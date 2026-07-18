# ComfyUI-Krea2BlockSwap. Dynamic block swap for the Krea 2 image model.
# Copyright (C) 2026 - olob0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import gc
import logging

import torch
import comfy.model_management as mm

log = logging.getLogger(__name__)


def _module_mem_mb(module):
    total = 0
    for p in module.parameters():
        if p.data is not None:
            total += p.nelement() * p.element_size()
    return total / (1024 * 1024)


def _find_blocks(diffusion_model):
    """Locate the transformer block ModuleList inside the Krea 2 diffusion model.

    Krea 2 (comfy/ldm/krea2/model.py -> SingleStreamDiT) exposes its stack of
    SingleStreamBlock modules as `.blocks`. The extra names are fallbacks in case a
    future ComfyUI release renames the attribute.
    """
    for attr in ("blocks", "transformer_blocks", "single_blocks"):
        blocks = getattr(diffusion_model, attr, None)
        if blocks is not None and len(blocks) > 0:
            return blocks, attr
    raise AttributeError(
        "Could not find the transformer block ModuleList on the Krea 2 diffusion "
        "model (expected '.blocks'). The architecture may have changed."
    )


class _SwapManager:
    """Keeps the first N transformer blocks in system RAM and streams them through VRAM.

    A swappable block lives on the offload device (CPU) while idle. A forward pre-hook
    moves it to the compute device (GPU) right before its forward pass, and a forward
    hook moves it back afterwards. This means every block still executes on the GPU with
    its original weights in their original order, so the output is bit-identical to a run
    without swapping. The only cost is the PCIe transfer time per block per step.
    """

    def __init__(self, blocks, num_swap, main_device, offload_device, use_non_blocking):
        self.blocks = blocks
        self.main_device = main_device
        self.offload_device = offload_device
        self.non_blocking = use_non_blocking
        self.hook_handles = []
        self.offloaded_mb = 0
        # Blocks [0, num_swap) are swappable; the rest stay resident on the GPU
        self.swap_ids = set(range(min(num_swap, len(blocks))))

    def _pre_hook(self, module, args):
        module.to(self.main_device, non_blocking=self.non_blocking)
        return None

    def _post_hook(self, module, args, output):
        module.to(self.offload_device, non_blocking=self.non_blocking)
        return output

    def apply(self):
        """Move swappable blocks to CPU, register their hooks, and pin the rest to GPU.

        Called once per sampling run. Any hooks left over from a run that failed before
        cleanup are removed first, so blocks are never double-hooked.
        """
        if self.hook_handles:
            self.remove()

        offloaded_mb = 0
        for i, block in enumerate(self.blocks):
            if i in self.swap_ids:
                block.to(self.offload_device, non_blocking=self.non_blocking)
                offloaded_mb += _module_mem_mb(block)
                self.hook_handles.append(block.register_forward_pre_hook(self._pre_hook))
                self.hook_handles.append(block.register_forward_hook(self._post_hook))
            else:
                block.to(self.main_device, non_blocking=self.non_blocking)

        mm.soft_empty_cache()
        gc.collect()
        self.offloaded_mb = offloaded_mb

    def remove(self):
        """Detach all hooks. Blocks keep whatever device they were last moved to."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []


class Krea2BlockSwap:
    """ComfyUI node: dynamic block swap for Krea 2.

    Offloads a chosen number of transformer blocks to system RAM to fit the model on
    GPUs with limited VRAM. Output quality is unaffected: swapping only changes where a
    block's weights live between forward passes, not the computation itself.

    Wire this node between the model loader and the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Loaded Krea 2 model."}),
                "blocks_to_swap": ("INT", {
                    "default": 8, "min": 0, "max": 48, "step": 1,
                    "tooltip": "Number of transformer blocks (starting from block 0) "
                               "kept in RAM while idle. Higher = less VRAM, slower.",
                }),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Asynchronous CPU <-> GPU transfers. Recommended on; turn "
                               "off only if you hit instability.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "olob0/Krea2"
    TITLE = "Krea2 Block Swap"

    def apply(self, model, blocks_to_swap, use_non_blocking=True):
        new_model = model.clone()

        blocks, attr = _find_blocks(new_model.model.diffusion_model)

        manager = _SwapManager(
            blocks,
            blocks_to_swap,
            mm.get_torch_device(),
            mm.unet_offload_device(),
            use_non_blocking,
        )

        logged = {"done": False}

        def unet_wrapper(model_function, kwargs):
            # Hooks are applied for the duration of the model call and removed right
            # after, so nothing is left patched on the shared module between runs.
            manager.apply()
            if not logged["done"]:
                log.info(
                    "Krea2BlockSwap: swapping %d/%d blocks (~%.0f MB kept out of VRAM "
                    "while idle).",
                    len(manager.swap_ids), len(blocks), manager.offloaded_mb,
                )
                logged["done"] = True
            try:
                return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            finally:
                manager.remove()

        new_model.set_model_unet_function_wrapper(unet_wrapper)
        log.info("Krea2BlockSwap: attached to '%s' (%d blocks).", attr, len(blocks))
        return (new_model,)
