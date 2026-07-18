from .block_swap import Krea2BlockSwap

NODE_CLASS_MAPPINGS = {
    "Krea2BlockSwap": Krea2BlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2BlockSwap": "Krea2 Block Swap",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
