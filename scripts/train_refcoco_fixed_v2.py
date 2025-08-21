# scripts/train_refcoco_fixed_v2.py
"""
修復版RefCOCO訓練腳本
解決空間推理中的張量尺寸不匹配問題
"""

import logging
import os
from pathlib import Path
from typing import Optional

import draccus
import torch
#from draccus import dataclass
from dataclasses import dataclass
from cobra.conf import ModelConfig
from cobra.models import load_vlm
from cobra.models.vlms.cobra_spatial_fixed import CobraSpatialFixed
from cobra.training import Metrics
from cobra.training.strategies import LoadStrategy
from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset
from cobra.preprocessing.multimodal_dataloaders import MultimodalDataLoader


@dataclass
class FixedRefCOCOTrainConfig:
    """修復版RefCOCO訓練配置"""
    
    # Model & Training Configuration
    model: ModelConfig = ModelConfig()
    refcoco_data_dir: Path = Path("./data/refcoco")
    max_samples: Optional[int] = None
    num_epochs: int = 2
    run_id: str = "refcoco-spatial-fixed"
    use_real_refcoco_data: bool = True
    
    # Spatial Reasoning Configuration
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "adaptive"
    debug_tensor_shapes: bool = True
    
    # Training Parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    
    # GPU Memory Optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True


def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%m/%d [%H:%M:%S]'
    )
    return logging.getLogger(__name__)


def create_fixed_refcoco_dataset(cfg: FixedRefCOCOTrainConfig, split: str = "train"):
    """創建修復版RefCOCO數據集"""
    
    logger = logging.getLogger(__name__)
    
    # 檢查數據文件
    if cfg.use_real_refcoco_data:
        annotations_file = cfg.refcoco_data_dir / "refcoco.json"
        if not annotations_file.exists():
            logger.warning(f"真實數據文件不存在: {annotations_file}")
            cfg.use_real_refcoco_data = False
    
    if not cfg.use_real_refcoco_data:
        # 創建合成數據
        logger.info("創建合成RefCOCO數據...")
        synthetic_data = create_synthetic_refcoco_data(cfg.max_samples or 100)
        return SyntheticRefCOCODataset(synthetic_data)
    
    # 使用真實數據
    logger.info(f"載入真實RefCOCO數據: {annotations_file}")
    dataset = RefCOCODataset(
        annotation_file=annotations_file,
        images_dir=cfg.refcoco_data_dir / "images",
        split=split,
        max_samples=cfg.max_samples
    )
    
    return dataset


def create_synthetic_refcoco_data(num_samples: int):
    """創建合成RefCOCO數據用於測試"""
    
    import random
    
    synthetic_data = []
    expressions = [
        "the red car on the left",
        "a person in blue shirt",
        "the white building in the center",
        "a dog running in the park",
        "the yellow flower in the garden"
    ]
    
    for i in range(num_samples):
        # 創建隨機邊界框
        x = random.uniform(0, 400)
        y = random.uniform(0, 300)
        w = random.uniform(50, 200)
        h = random.uniform(50, 150)
        
        example = {
            "image_id": f"synthetic_{i}",
            "image_path": "synthetic_image.jpg",  # 將使用隨機圖像
            "expression": random.choice(expressions),
            "bbox": [x, y, w, h],
            "image_width": 640,
            "image_height": 480
        }
        synthetic_data.append(example)
    
    return synthetic_data


class SyntheticRefCOCODataset:
    """合成RefCOCO數據集用於測試"""
    
    def __init__(self, synthetic_data):
        self.examples = synthetic_data
        
        # 創建合成圖像變換
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 創建隨機圖像
        import torch
        from PIL import Image
        import numpy as np
        
        # 生成隨機RGB圖像
        random_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image = Image.fromarray(random_image)
        pixel_values = self.image_transform(image)
        
        # 創建輸入文本
        prompt = f"Please locate: {example['expression']}"
        
        return {
            "pixel_values": pixel_values,
            "text": prompt,
            "bbox": torch.tensor(example["bbox"], dtype=torch.float32),
            "image_size": torch.tensor([example["image_width"], example["image_height"]], dtype=torch.float32)
        }
    
    def get_modality_lengths(self):
        """返回模態長度信息"""
        return [(True, 20) for _ in range(len(self.examples))]  # 所有樣本都是多模態


def create_fixed_spatial_model(cfg: FixedRefCOCOTrainConfig):
    """創建修復版空間推理模型"""
    
    logger = logging.getLogger(__name__)
    logger.info("載入模型組件...")
    
    # 基於配置創建模型
    if cfg.model.type == "cobra-refcoco-lora+3b":
        model_config = ModelConfig(
            model_id="cobra-spatial-fixed+3b",
            arch_specifier="no-align+fused-gelu-mlp",
            vision_backbone_id="dinosiglip-vit-so-384px",
            llm_backbone_id="mamba-2.8b-zephyr",
            image_resize_strategy="resize-naive",
            llm_max_length=2048,
        )
    else:
        model_config = cfg.model
    
    # 使用修復版的空間推理模型
    vlm = CobraSpatialFixed.from_pretrained(
        model_config.model_id,
        enable_spatial_reasoning=cfg.enable_spatial_reasoning,
        spatial_module_type=cfg.spatial_module_type,
        debug_tensor_shapes=cfg.debug_tensor_shapes,
    )
    
    # 檢查模型參數
    total_params = sum(p.numel() for p in vlm.parameters())
    trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    
    logger.info(f"總參數數量: {total_params:,}")
    logger.info(f"可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return vlm


def train_with_tensor_shape_debugging(vlm, dataloader, cfg: FixedRefCOCOTrainConfig):
    """帶張量形狀調試的訓練函數"""
    
    logger = logging.getLogger(__name__)
    
    # 設置優化器
    optimizer = torch.optim.AdamW(
        vlm.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    vlm.train()
    
    for epoch in range(cfg.num_epochs):
        logger.info(f"開始 Epoch {epoch + 1}/{cfg.num_epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 調試：打印批次信息
                if cfg.debug_tensor_shapes:
                    logger.info(f"批次 {batch_idx}: ")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.shape}")
                        else:
                            logger.info(f"  {key}: {type(value)}")
                
                # 前向傳播
                outputs = vlm(**batch)
                loss = outputs.loss
                
                # 反向傳播
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(vlm.parameters(), cfg.max_grad_norm)
                    
                    # 優化器步驟
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"  批次 {batch_idx}, 損失: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx} 處理失敗: {e}")
                if cfg.debug_tensor_shapes:
                    import traceback
                    traceback.print_exc()
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} 平均損失: {avg_loss:.4f}")


@draccus.wrap()
def train_fixed_refcoco(cfg: FixedRefCOCOTrainConfig):
    """修復版RefCOCO訓練主函數"""
    
    logger = setup_logging()
    logger.info("🚀 開始修復版RefCOCO訓練")
    logger.info(f"模型: {cfg.model.type}")
    logger.info(f"樣本數: {cfg.max_samples}")
    logger.info(f"真實數據: {cfg.use_real_refcoco_data}")
    logger.info(f"空間推理: {cfg.enable_spatial_reasoning}")
    logger.info(f"調試模式: {cfg.debug_tensor_shapes}")
    
    try:
        # 創建數據集
        logger.info("創建修復版RefCOCO數據集...")
        dataset = create_fixed_refcoco_dataset(cfg)
        logger.info(f"成功載入RefCOCO數據: {len(dataset)} 條目")
        
        # 創建模型
        vlm = create_fixed_spatial_model(cfg)
        
        # 創建數據加載器
        dataloader = MultimodalDataLoader(
            [dataset],
            collate_fn=vlm.get_collate_fn(),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0  # 單線程避免問題
        )
        
        # GPU內存檢查
        if torch.cuda.is_available():
            device = torch.device("cuda")
            vlm = vlm.to(device)
            
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"GPU內存 - 已分配: {allocated:.1f}GB, 已緩存: {cached:.1f}GB, 總計: {total:.1f}GB")
        
        # 開始訓練
        logger.info("🎯 開始修復版訓練...")
        train_with_tensor_shape_debugging(vlm, dataloader, cfg)
        
        logger.info("✅ 訓練完成!")
        
        # 保存模型
        save_path = Path(f"./runs/{cfg.run_id}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': vlm.state_dict(),
            'config': cfg,
            'epoch': cfg.num_epochs
        }, save_path / "spatial_fixed_model.pt")
        
        logger.info(f"模型已保存到: {save_path}")
        
    except Exception as e:
        logger.error(f"訓練過程錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    train_fixed_refcoco()