"""
registry.py

Exhaustive list of pretrained VLMs (with full descriptions / links to corresponding names and sections of paper).
"""

# === Pretrained Model Registry ===
# fmt: off
MODEL_REGISTRY = {
    "cobra+3b": {
        "model_id": "cobra+3b",
        "names": ["Cobra-DINOSigLIP 3B"],
        "description": {
            "name": "Cobra 3B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
    "cobra-blip2+3b": {
        "model_id": "cobra-blip2+3b",
        "names": ["Cobra-BLIP2 3B"],
        "description": {
            "name": "Cobra BLIP-2 3B",
            "optimization_procedure": "single-stage",
            "visual_representation": "BLIP-2 ViT-g @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
    "cobra-dinoblip2+3b": {
        "model_id": "cobra-dinoblip2+3b",
        "names": ["Cobra-DinoBLIP2 3B"],
        "description": {
            "name": "Cobra DINOv2 + BLIP-2 3B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + BLIP-2 ViT-g @ 384px",
            "image_processing": "Naive Resize", 
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
    "cobra-dinoblip2-l+3b": {
        "model_id": "cobra-dinoblip2-l+3b",
        "names": ["Cobra-DinoBLIP2-L 3B"],
        "description": {
            "name": "Cobra DINOv2 + BLIP-2 L 3B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + BLIP-2 ViT-L @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Mamba 2.8B Zephyr", 
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
}

# === 現在添加新的空間推理模型 ===
MODEL_REGISTRY.update({
    "cobra-refcoco-lora+3b": {
        "model_id": "cobra-refcoco-lora+3b",
        "names": ["Cobra RefCOCO LoRA 3B"],
        "description": {
            "name": "Cobra RefCOCO LoRA 3B",
            "optimization_procedure": "spatial-lora-finetune",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Multi-Directional Spatial Scanning + Naive Resize",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["RefCOCO", "RefCOCO+", "RefCOCOg"],
            "spatial_reasoning": True,
            "multi_directional_scan": True,
            "lora_enabled": True,
            "train_epochs": 8,
        },
    },
    "cobra-spatial-refcoco+3b": {
        "model_id": "cobra-spatial-refcoco+3b",
        "names": ["Cobra Spatial RefCOCO 3B"],
        "description": {
            "name": "Cobra Spatial RefCOCO 3B",
            "optimization_procedure": "spatial-finetune",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Multi-Directional Spatial Scanning + Naive Resize",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["RefCOCO", "RefCOCO+", "RefCOCOg"],
            "spatial_reasoning": True,
            "multi_directional_scan": True,
            "train_epochs": 5,
        },
    }
})

# Add MultiMamba variant emphasizing spatial reasoning and LoRA
MODEL_REGISTRY.update({
    "multimamba-spatial-lora+3b": {
        "model_id": "multimamba-spatial-lora+3b",
        "names": ["MultiMamba Spatial LoRA 3B"],
        "description": {
            "name": "MultiMamba Spatial LoRA 3B",
            "optimization_procedure": "spatial-lora-finetune",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Mamba Spatial Scanner (multi-directional)",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["RefCOCO", "RefCOCO+", "RefCOCOg"],
            "spatial_reasoning": True,
            "multi_directional_scan": True,
            "lora_enabled": True,
            "train_epochs": 6,
        },
    },
})

# Build Global Registry (Model ID, Name) -> Metadata
GLOBAL_REGISTRY = {name: v for k, v in MODEL_REGISTRY.items() for name in [k] + v["names"]}

# fmt: on