from .download import convert_to_jpg, download_extract
from .materialize import get_dataset_and_collator

# 嘗試導入RefCOCO相關功能
try:
    from .materialize_refcoco import (
        get_dataset_and_collator_refcoco,
        get_spatial_enhanced_dataset_and_collator,
        create_refcoco_training_setup
    )
    REFCOCO_AVAILABLE = True
    __all__ = [
        'convert_to_jpg', 'download_extract', 'get_dataset_and_collator',
        'get_dataset_and_collator_refcoco', 'get_spatial_enhanced_dataset_and_collator',
        'create_refcoco_training_setup'
    ]
except ImportError:
    REFCOCO_AVAILABLE = False
    __all__ = ['convert_to_jpg', 'download_extract', 'get_dataset_and_collator']