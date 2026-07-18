# ComfyUI-Krea2BlockSwap

Dynamic block swap for the **Krea 2** image model. Offloads transformer blocks to system RAM so the model fits on GPUs with limited VRAM, **without any quality loss**.

## How it works

Krea 2 (`comfy/ldm/krea2/model.py` -> `SingleStreamDiT`) keeps its heavy transformer blocks in `diffusion_model.blocks`. During sampling, this node registers per-block hooks:

- a **forward pre-hook** moves a block to the GPU right before its forward pass
- a **forward hook** moves it back to the CPU right after

Swappable blocks live in RAM while idle but always **execute on the GPU** with their original weights in their original order. Moving a tensor between devices does not change its values, so the output is **bit-identical** to a run without swapping. **The only cost is PCIe transfer time**.

This is different from static offload approaches that leave blocks parked on the CPU. Those either run on the CPU (very slow) or hit device-mismatch errors.

## Installation

Copy or clone the `ComfyUI-Krea2BlockSwap` folder into `ComfyUI/custom_nodes/` and restart ComfyUI. You also can use ComfyUI Manager to install. No dependencies required.

## Usage

1. Load Krea 2 model. 
2. Insert **Krea2 Block Swap** between the loader and the `KSampler`.
3. Set `blocks_to_swap`. Start at 8 and raise it until it fits. Each additional block frees more VRAM while idle but adds one CPU <-> GPU transfer per step.

## Parameters

- **blocks_to_swap** (0–48): how many blocks, starting from block 0, stay in RAM while idle.
- **use_non_blocking** (bool): asynchronous transfers. Leave on. Turn off only if you hit instability.

## Notes

- Works with any quantization (int8_convrot, nvfp4, mxfp8, fp8) — swapping acts on the loaded modules regardless of precision.
- The Qwen3-VL text encoder is unaffected.
- If a future ComfyUI release renames the block attribute, adjust `_find_blocks` in `block_swap.py`.

## License

GPL-3.0. See [LICENSE](LICENSE).
