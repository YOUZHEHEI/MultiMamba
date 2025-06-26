from .datasets import DatasetConfig, DatasetRegistry
from .models import ModelConfig, ModelRegistry

# 嘗試導入空間模型配置
try:
    from .models_spatial import SpatialModelConfig, SpatialModelRegistry
    SPATIAL_MODELS_AVAILABLE = True
    __all__ = ['DatasetConfig', 'DatasetRegistry', 'ModelConfig', 'ModelRegistry', 
               'SpatialModelConfig', 'SpatialModelRegistry']
except ImportError:
    SPATIAL_MODELS_AVAILABLE = False
    __all__ = ['DatasetConfig', 'DatasetRegistry', 'ModelConfig', 'ModelRegistry']

# 嘗試導入RefCOCO數據集配置
try:
    from .refcoco_datasets import RefCOCODatasetConfig, RefCOCODatasetRegistry
    __all__.extend(['RefCOCODatasetConfig', 'RefCOCODatasetRegistry'])
except ImportError:
    pass