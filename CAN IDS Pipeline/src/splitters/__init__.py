from .default import DefaultSplitter
from .threeway import ThreeWaySplitter
from config import *
SPLIT_REGISTRY = {
    "default": DefaultSplitter,
    "three": ThreeWaySplitter,
}

def get_splitter(input_dir, mode, feature_extractor="PixNet", **kwargs):
    if not SPLIT: 
        return None
    splitter_cls = SPLIT_REGISTRY.get(mode)
    if splitter_cls is None:
        raise ValueError(f"Unknown split mode: {mode}")
    return splitter_cls(input_dir, feature_extractor, **kwargs).split()
