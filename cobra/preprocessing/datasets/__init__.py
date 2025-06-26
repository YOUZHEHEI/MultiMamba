from .datasets import AlignDataset, FinetuneDataset

# 嘗試導入RefCOCO數據集
try:
    from .refcoco_dataset import RefCOCODataset
    __all__ = ['AlignDataset', 'FinetuneDataset', 'RefCOCODataset']
except ImportError:
    __all__ = ['AlignDataset', 'FinetuneDataset']

# 嘗試導入混合數據集
try:
    from .mixed_spatial_dataset import MixedSpatialDataset, BalancedMixedSpatialDataset
    __all__.extend(['MixedSpatialDataset', 'BalancedMixedSpatialDataset'])
except ImportError:
    pass