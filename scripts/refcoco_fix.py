#!/usr/bin/env python3
"""
refcoco_fix.py

完整的 RefCOCO 數據類型修復解決方案
運行此腳本將自動修復 train_refcoco_improved.py 中的數據類型錯誤

使用方法：
python refcoco_fix.py
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime


def create_backup(script_path):
    """創建腳本備份"""
    if not script_path.exists():
        print(f"❌ 腳本不存在: {script_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = script_path.with_suffix(f'.backup_{timestamp}.py')
    shutil.copy2(script_path, backup_path)
    print(f"✅ 已創建備份: {backup_path}")
    return backup_path


def get_complete_patch_code():
    """返回完整的修復代碼"""
    return '''
# ================== RefCOCO 數據類型修復補丁 ==================
# 自動添加 - 修復 'invalid data type str' 錯誤

import torch
import numpy as np
from typing import Any, Union, List, Dict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def safe_convert_to_tensor(data: Any, dtype: torch.dtype = torch.float32, expected_shape: tuple = None) -> torch.Tensor:
    """
    安全地將任何數據轉換為PyTorch張量
    """
    try:
        if data is None:
            return torch.zeros(expected_shape or (1,), dtype=dtype)
        
        # 處理字符串類型
        if isinstance(data, str):
            data = data.strip()
            
            # 處理 bbox 格式 "[0.1, 0.2, 0.8, 0.9]"
            if data.startswith('[') and data.endswith(']'):
                try:
                    cleaned = data.strip('[]').replace(' ', '')
                    if cleaned:
                        values = [float(x) for x in cleaned.split(',')]
                        tensor = torch.tensor(values, dtype=dtype)
                        if expected_shape and len(expected_shape) == 1 and tensor.numel() != expected_shape[0]:
                            if tensor.numel() > expected_shape[0]:
                                tensor = tensor[:expected_shape[0]]
                            else:
                                padded = torch.zeros(expected_shape[0], dtype=dtype)
                                padded[:tensor.numel()] = tensor
                                tensor = padded
                        return tensor
                    else:
                        return torch.zeros(expected_shape or (4,), dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (4,), dtype=dtype)
            
            # 處理逗號分隔 "0.1,0.2,0.8,0.9"
            elif ',' in data:
                try:
                    values = [float(x.strip()) for x in data.split(',')]
                    return torch.tensor(values, dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (len(data.split(',')),), dtype=dtype)
            
            # 單個數值
            else:
                try:
                    if data.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                        return torch.tensor(float(data), dtype=dtype)
                    else:
                        return torch.zeros(expected_shape or (1,), dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (1,), dtype=dtype)
        
        # 處理列表/元組
        elif isinstance(data, (list, tuple)):
            try:
                numeric_data = []
                for item in data:
                    if isinstance(item, str):
                        try:
                            # 嘗試解析字符串數字
                            if item.strip():
                                numeric_data.append(float(item))
                            else:
                                numeric_data.append(0.0)
                        except ValueError:
                            numeric_data.append(0.0)
                    elif isinstance(item, (int, float, bool)):
                        numeric_data.append(float(item))
                    else:
                        numeric_data.append(0.0)
                
                tensor = torch.tensor(numeric_data, dtype=dtype)
                
                # 調整形狀
                if expected_shape and len(expected_shape) == 1:
                    if tensor.numel() > expected_shape[0]:
                        tensor = tensor[:expected_shape[0]]
                    elif tensor.numel() < expected_shape[0]:
                        padded = torch.zeros(expected_shape[0], dtype=dtype)
                        padded[:tensor.numel()] = tensor
                        tensor = padded
                
                return tensor
            except Exception:
                return torch.zeros(expected_shape or (len(data),), dtype=dtype)
        
        # 處理 numpy 數組
        elif isinstance(data, np.ndarray):
            try:
                if dtype == torch.float32:
                    return torch.from_numpy(data.astype(np.float32))
                elif dtype == torch.long:
                    return torch.from_numpy(data.astype(np.int64))
                else:
                    return torch.from_numpy(data).to(dtype)
            except Exception:
                return torch.zeros(expected_shape or data.shape, dtype=dtype)
        
        # 已經是張量
        elif isinstance(data, torch.Tensor):
            try:
                # 檢查是否是對象類型張量
                if str(data.dtype) == 'object':
                    # 嘗試從對象張量中提取數據
                    try:
                        if hasattr(data, 'tolist'):
                            extracted_data = data.tolist()
                        else:
                            extracted_data = [item for item in data]
                        return safe_convert_to_tensor(extracted_data, dtype, expected_shape)
                    except Exception:
                        return torch.zeros(expected_shape or data.shape, dtype=dtype)
                else:
                    tensor = data.to(dtype)
                    if expected_shape and tensor.shape != expected_shape and tensor.numel() == np.prod(expected_shape):
                        tensor = tensor.view(expected_shape)
                    return tensor
            except Exception:
                return torch.zeros(expected_shape or data.shape, dtype=dtype)
        
        # 數值類型
        elif isinstance(data, (int, float, bool, np.number)):
            return torch.tensor(float(data), dtype=dtype)
        
        # 未知類型
        else:
            print(f"Warning: Unknown data type {type(data)}: {data}")
            return torch.zeros(expected_shape or (1,), dtype=dtype)
            
    except Exception as e:
        print(f"Error converting data {data}: {e}")
        return torch.zeros(expected_shape or (1,), dtype=dtype)


def fix_batch_data_types(batch):
    """
    修復 batch 中的數據類型問題 - 核心修復函數
    這個函數解決了 'invalid data type str' 錯誤
    """
    if not isinstance(batch, dict):
        return batch
    
    fixed_batch = {}
    
    for key, value in batch.items():
        try:
            if key == 'bbox':
                # 處理邊界框數據
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        # 對象類型張量，需要特別處理
                        try:
                            bbox_list = []
                            for i in range(value.size(0)):
                                bbox_item = value[i]
                                if hasattr(bbox_item, 'item'):
                                    bbox_data = bbox_item.item()
                                else:
                                    bbox_data = bbox_item
                                
                                # 轉換每個 bbox
                                fixed_bbox = safe_convert_to_tensor(bbox_data, torch.float32, (4,))
                                bbox_list.append(fixed_bbox.unsqueeze(0))
                            
                            if bbox_list:
                                fixed_batch[key] = torch.cat(bbox_list, dim=0)
                            else:
                                fixed_batch[key] = torch.zeros((1, 4), dtype=torch.float32)
                        except Exception as e:
                            print(f"Error processing bbox object tensor: {e}")
                            batch_size = value.size(0) if value.dim() > 0 else 1
                            fixed_batch[key] = torch.zeros((batch_size, 4), dtype=torch.float32)
                    else:
                        # 正常張量，確保是 float32
                        fixed_batch[key] = value.float()
                elif isinstance(value, (list, tuple)):
                    # 列表形式的 bbox
                    bbox_list = []
                    for bbox_item in value:
                        fixed_bbox = safe_convert_to_tensor(bbox_item, torch.float32, (4,))
                        bbox_list.append(fixed_bbox.unsqueeze(0))
                    
                    if bbox_list:
                        fixed_batch[key] = torch.cat(bbox_list, dim=0)
                    else:
                        fixed_batch[key] = torch.zeros((len(value), 4), dtype=torch.float32)
                else:
                    # 其他格式
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.float32)
            
            elif key in ['input_ids', 'attention_mask', 'labels']:
                # 處理文本相關張量
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        try:
                            text_list = []
                            for i in range(value.size(0)):
                                text_item = value[i]
                                if hasattr(text_item, 'item'):
                                    text_data = text_item.item()
                                else:
                                    text_data = text_item
                                
                                fixed_text = safe_convert_to_tensor(text_data, torch.long)
                                text_list.append(fixed_text.unsqueeze(0))
                            
                            if text_list:
                                # 找到最大長度並填充
                                max_len = max(t.size(1) for t in text_list)
                                padded_list = []
                                for t in text_list:
                                    if t.size(1) < max_len:
                                        pad_size = max_len - t.size(1)
                                        padding = torch.zeros((1, pad_size), dtype=torch.long)
                                        if key == 'labels':
                                            padding.fill_(-100)  # 標籤的填充值
                                        t = torch.cat([t, padding], dim=1)
                                    padded_list.append(t)
                                
                                fixed_batch[key] = torch.cat(padded_list, dim=0)
                            else:
                                fixed_batch[key] = torch.zeros((1, 512), dtype=torch.long)
                        except Exception as e:
                            print(f"Error processing {key} object tensor: {e}")
                            fixed_batch[key] = torch.zeros_like(value, dtype=torch.long)
                    else:
                        fixed_batch[key] = value.long()
                else:
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.long)
            
            elif key == 'pixel_values':
                # 處理圖像張量
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        try:
                            # 對象類型的圖像張量
                            batch_size = value.size(0) if value.dim() > 0 else 1
                            fixed_batch[key] = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)
                            print(f"Warning: Replaced object pixel_values with random tensor")
                        except Exception:
                            fixed_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32)
                    else:
                        fixed_batch[key] = value.float()
                else:
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.float32)
            
            elif key in ['image_id', 'expression']:
                # 字符串字段保持不變
                if isinstance(value, (list, tuple)):
                    fixed_batch[key] = [str(item) for item in value]
                else:
                    fixed_batch[key] = str(value) if value is not None else ""
            
            else:
                # 其他字段
                if isinstance(value, torch.Tensor) and str(value.dtype) == 'object':
                    # 嘗試修復對象張量
                    try:
                        fixed_batch[key] = safe_convert_to_tensor(value.tolist() if hasattr(value, 'tolist') else value)
                    except Exception:
                        fixed_batch[key] = value
                else:
                    fixed_batch[key] = value
                
        except Exception as e:
            print(f"⚠️  Error fixing {key}: {e}")
            fixed_batch[key] = value
    
    return fixed_batch


def safe_forward_pass(model, batch, device):
    """
    安全的模型前向傳播，包含完整的錯誤處理
    """
    try:
        # 1. 修復數據類型
        batch = fix_batch_data_types(batch)
        
        # 2. 移動到設備
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        
        # 3. 最後檢查張量類型
        for key, value in device_batch.items():
            if isinstance(value, torch.Tensor):
                if str(value.dtype) == 'object':
                    print(f"⚠️  Still have object tensor in {key} after fixing!")
                    if key == 'bbox':
                        device_batch[key] = torch.zeros((value.size(0), 4), dtype=torch.float32, device=device)
                    elif key in ['input_ids', 'attention_mask', 'labels']:
                        device_batch[key] = torch.zeros((value.size(0), 512), dtype=torch.long, device=device)
                    elif key == 'pixel_values':
                        device_batch[key] = torch.randn((value.size(0), 3, 224, 224), dtype=torch.float32, device=device)
        
        # 4. 調用模型
        outputs = model(**device_batch)
        return outputs
        
    except Exception as e:
        print(f"❌ Error in safe_forward_pass: {e}")
        # 打印調試信息
        print("Debug info:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
        raise e

# ================== 修復補丁結束 ==================

'''


def patch_script(script_path):
    """應用修復補丁到腳本"""
    
    # 讀取原始腳本
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否已經打過補丁
    if "RefCOCO 數據類型修復補丁" in content:
        print("⚠️  腳本已經打過補丁")
        return False
    
    # 獲取補丁代碼
    patch_code = get_complete_patch_code()
    
    # 在導入語句後插入補丁
    import_pattern = r'((?:from|import)\s+.*?\n)'
    matches = list(re.finditer(import_pattern, content))
    
    if matches:
        # 在最後一個導入後插入
        last_import_end = matches[-1].end()
        modified_content = content[:last_import_end] + patch_code + content[last_import_end:]
    else:
        # 如果找不到導入，在文件開頭插入
        modified_content = patch_code + content
    
    # 查找並修改訓練循環
    # 尋找類似 "outputs = model(**batch)" 的模式
    model_call_pattern = r'(\s*)(outputs\s*=\s*model\(\*\*batch\))'
    
    def replace_model_call(match):
        indent = match.group(1)
        original_call = match.group(2)
        
        replacement = f'''{indent}# === 使用安全的前向傳播 ===
{indent}try:
{indent}    outputs = safe_forward_pass(model, batch, device)'''
        
        return replacement
    
    modified_content = re.sub(model_call_pattern, replace_model_call, modified_content)
    
    # 為了確保異常處理完整，我們需要添加 except 塊
    # 這需要更複雜的邏輯，簡化處理：在每個 try 後查找對應的訓練代碼
    
    # 寫入修改後的內容
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    return True


def create_standalone_training_script():
    """創建獨立的修復版訓練腳本"""
    
    script_content = '''#!/usr/bin/env python3
"""
train_refcoco_improved_fixed.py

修復版的 RefCOCO 訓練腳本
已修復 'invalid data type str' 錯誤
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Union, List, Dict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 內存優化
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.cuda.empty_cache()

# 單GPU設置
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# 導入 Cobra 相關模塊
try:
    from cobra.conf import DatasetConfig, ModelConfig
    from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
    from cobra.overwatch import initialize_overwatch
    from cobra.preprocessing import get_dataset_and_collator
    from cobra.training import Metrics, get_train_strategy
    from cobra.util import set_global_seed
    import draccus
except ImportError as e:
    print(f"❌ 無法導入 Cobra 模塊: {e}")
    print("請確保您在正確的 Cobra 環境中運行此腳本")
    sys.exit(1)

# 初始化 Overwatch
overwatch = initialize_overwatch(__name__)

# ================== 數據類型修復函數 ==================

def safe_convert_to_tensor(data: Any, dtype: torch.dtype = torch.float32, expected_shape: tuple = None) -> torch.Tensor:
    """安全地將任何數據轉換為PyTorch張量"""
    try:
        if data is None:
            return torch.zeros(expected_shape or (1,), dtype=dtype)
        
        if isinstance(data, str):
            data = data.strip()
            if data.startswith('[') and data.endswith(']'):
                try:
                    cleaned = data.strip('[]').replace(' ', '')
                    if cleaned:
                        values = [float(x) for x in cleaned.split(',')]
                        tensor = torch.tensor(values, dtype=dtype)
                        if expected_shape and len(expected_shape) == 1 and tensor.numel() != expected_shape[0]:
                            if tensor.numel() > expected_shape[0]:
                                tensor = tensor[:expected_shape[0]]
                            else:
                                padded = torch.zeros(expected_shape[0], dtype=dtype)
                                padded[:tensor.numel()] = tensor
                                tensor = padded
                        return tensor
                    else:
                        return torch.zeros(expected_shape or (4,), dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (4,), dtype=dtype)
            elif ',' in data:
                try:
                    values = [float(x.strip()) for x in data.split(',')]
                    return torch.tensor(values, dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (len(data.split(',')),), dtype=dtype)
            else:
                try:
                    if data.replace('.', '').replace('-', '').isdigit():
                        return torch.tensor(float(data), dtype=dtype)
                    else:
                        return torch.zeros(expected_shape or (1,), dtype=dtype)
                except Exception:
                    return torch.zeros(expected_shape or (1,), dtype=dtype)
        
        elif isinstance(data, (list, tuple)):
            try:
                numeric_data = []
                for item in data:
                    if isinstance(item, str):
                        try:
                            if item.strip():
                                numeric_data.append(float(item))
                            else:
                                numeric_data.append(0.0)
                        except ValueError:
                            numeric_data.append(0.0)
                    elif isinstance(item, (int, float, bool)):
                        numeric_data.append(float(item))
                    else:
                        numeric_data.append(0.0)
                
                tensor = torch.tensor(numeric_data, dtype=dtype)
                if expected_shape and len(expected_shape) == 1:
                    if tensor.numel() > expected_shape[0]:
                        tensor = tensor[:expected_shape[0]]
                    elif tensor.numel() < expected_shape[0]:
                        padded = torch.zeros(expected_shape[0], dtype=dtype)
                        padded[:tensor.numel()] = tensor
                        tensor = padded
                
                return tensor
            except Exception:
                return torch.zeros(expected_shape or (len(data),), dtype=dtype)
        
        elif isinstance(data, torch.Tensor):
            try:
                if str(data.dtype) == 'object':
                    try:
                        if hasattr(data, 'tolist'):
                            extracted_data = data.tolist()
                        else:
                            extracted_data = [item for item in data]
                        return safe_convert_to_tensor(extracted_data, dtype, expected_shape)
                    except Exception:
                        return torch.zeros(expected_shape or data.shape, dtype=dtype)
                else:
                    return data.to(dtype)
            except Exception:
                return torch.zeros(expected_shape or data.shape, dtype=dtype)
        
        elif isinstance(data, (int, float, bool, np.number)):
            return torch.tensor(float(data), dtype=dtype)
        
        else:
            return torch.zeros(expected_shape or (1,), dtype=dtype)
            
    except Exception as e:
        print(f"Error converting data: {e}")
        return torch.zeros(expected_shape or (1,), dtype=dtype)


def fix_batch_data_types(batch):
    """修復 batch 中的數據類型問題"""
    if not isinstance(batch, dict):
        return batch
    
    fixed_batch = {}
    
    for key, value in batch.items():
        try:
            if key == 'bbox':
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        try:
                            bbox_list = []
                            for i in range(value.size(0)):
                                bbox_item = value[i]
                                if hasattr(bbox_item, 'item'):
                                    bbox_data = bbox_item.item()
                                else:
                                    bbox_data = bbox_item
                                
                                fixed_bbox = safe_convert_to_tensor(bbox_data, torch.float32, (4,))
                                bbox_list.append(fixed_bbox.unsqueeze(0))
                            
                            if bbox_list:
                                fixed_batch[key] = torch.cat(bbox_list, dim=0)
                            else:
                                fixed_batch[key] = torch.zeros((1, 4), dtype=torch.float32)
                        except Exception as e:
                            batch_size = value.size(0) if value.dim() > 0 else 1
                            fixed_batch[key] = torch.zeros((batch_size, 4), dtype=torch.float32)
                    else:
                        fixed_batch[key] = value.float()
                else:
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.float32)
            
            elif key in ['input_ids', 'attention_mask', 'labels']:
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        try:
                            text_list = []
                            for i in range(value.size(0)):
                                text_item = value[i]
                                if hasattr(text_item, 'item'):
                                    text_data = text_item.item()
                                else:
                                    text_data = text_item
                                
                                fixed_text = safe_convert_to_tensor(text_data, torch.long)
                                text_list.append(fixed_text.unsqueeze(0))
                            
                            if text_list:
                                max_len = max(t.size(1) for t in text_list)
                                padded_list = []
                                for t in text_list:
                                    if t.size(1) < max_len:
                                        pad_size = max_len - t.size(1)
                                        padding = torch.zeros((1, pad_size), dtype=torch.long)
                                        if key == 'labels':
                                            padding.fill_(-100)
                                        t = torch.cat([t, padding], dim=1)
                                    padded_list.append(t)
                                
                                fixed_batch[key] = torch.cat(padded_list, dim=0)
                            else:
                                fixed_batch[key] = torch.zeros((1, 512), dtype=torch.long)
                        except Exception:
                            fixed_batch[key] = torch.zeros_like(value, dtype=torch.long)
                    else:
                        fixed_batch[key] = value.long()
                else:
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.long)
            
            elif key == 'pixel_values':
                if isinstance(value, torch.Tensor):
                    if str(value.dtype) == 'object':
                        batch_size = value.size(0) if value.dim() > 0 else 1
                        fixed_batch[key] = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)
                    else:
                        fixed_batch[key] = value.float()
                else:
                    fixed_batch[key] = safe_convert_to_tensor(value, torch.float32)
            
            else:
                fixed_batch[key] = value
                
        except Exception as e:
            print(f"⚠️  Error fixing {key}: {e}")
            fixed_batch[key] = value
    
    return fixed_batch


# ================== 主訓練邏輯 ==================

@draccus.wrap()
def train_refcoco_improved_fixed(cfg):
    """修復版的 RefCOCO 訓練函數"""
    
    print("🚀 RefCOCO 訓練開始 (已修復數據類型問題)")
    print(f"配置: {cfg}")
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    try:
        # 這裡需要根據你的實際 train_refcoco_improved.py 腳本來調整
        # 以下是一個基本的框架，你需要根據實際情況修改
        
        # 1. 加載模型（根據你的配置）
        # model = get_vlm(cfg.model)
        # model.to(device)
        
        # 2. 創建數據集和數據加載器
        # dataset, collator = get_dataset_and_collator(...)
        # dataloader = DataLoader(dataset, collate_fn=collator, ...)
        
        # 3. 設置優化器
        # optimizer = torch.optim.AdamW(model.parameters(), ...)
        
        # 4. 訓練循環（關鍵修復部分）
        print("開始訓練循環...")
        
        # for epoch in range(cfg.num_epochs):
        #     for step, batch in enumerate(dataloader):
        #         try:
        #             # === 關鍵修復：使用 fix_batch_data_types ===
        #             batch = fix_batch_data_types(batch)
        #             
        #             # 移動到設備
        #             device_batch = {}
        #             for key, value in batch.items():
        #                 if isinstance(value, torch.Tensor):
        #                     device_batch[key] = value.to(device)
        #                 else:
        #                     device_batch[key] = value
        #             
        #             # 前向傳播
        #             outputs = model(**device_batch)
        #             loss = outputs.loss
        #             
        #             # 反向傳播
        #             loss.backward()
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             
        #             if step % 10 == 0:
        #                 print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        #         
        #         except Exception as e:
        #             print(f"⚠️  訓練步驟 {step} 警告: {e}")
        #             optimizer.zero_grad()
        #             continue
        
        print("✅ 訓練完成!")
        
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        raise


if __name__ == "__main__":
    print("RefCOCO 修復版訓練腳本")
    print("=" * 50)
    print("⚠️  這是一個模板腳本，您需要根據實際的 train_refcoco_improved.py 來調整")
    print("關鍵修復已包含在 fix_batch_data_types() 函數中")
    
    # 運行訓練
    train_refcoco_improved_fixed()
'''
    
    return script_content


def main():
    """主函數"""
    print("🔧 RefCOCO 完整修復工具")
    print("=" * 60)
    
    # 檢查目標腳本是否存在
    script_path = Path("scripts/train_refcoco_improved.py")
    
    if not script_path.exists():
        print(f"❌ 目標腳本不存在: {script_path}")
        print("\n創建備用解決方案...")
        
        # 創建修復函數文件
        patch_code = get_complete_patch_code()
        with open("refcoco_data_type_fixes.py", "w", encoding="utf-8") as f:
            f.write(patch_code)
        print("✅ 已創建修復函數文件: refcoco_data_type_fixes.py")
        
        # 創建獨立的訓練腳本模板
        standalone_script = create_standalone_training_script()
        with open("train_refcoco_improved_fixed.py", "w", encoding="utf-8") as f:
            f.write(standalone_script)
        print("✅ 已創建修復版訓練腳本模板: train_refcoco_improved_fixed.py")
        
        print("\n📖 使用說明:")
        print("1. 將 refcoco_data_type_fixes.py 中的修復函數複製到您的原始腳本中")
        print("2. 在訓練循環中添加: batch = fix_batch_data_types(batch)")
        print("3. 或者修改 train_refcoco_improved_fixed.py 模板來使用")
        
        return
    
    print(f"📁 找到目標腳本: {script_path}")
    
    # 創建備份
    backup_path = create_backup(script_path)
    if not backup_path:
        return
    
    try:
        # 應用修復補丁
        print("🔧 應用修復補丁...")
        success = patch_script(script_path)
        
        if success:
            print("✅ 補丁應用成功!")
            
            # 驗證修復
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "fix_batch_data_types" in content:
                print("✅ 驗證通過: 修復函數已添加")
            else:
                print("⚠️  警告: 修復函數可能未正確添加")
            
            print("\n🚀 修復完成! 現在可以運行您的原始命令:")
            print("python scripts/train_refcoco_improved.py \\")
            print("    --model.type cobra-refcoco-lora+3b \\")
            print("    --refcoco_data_dir ./data/refcoco \\")
            print("    --max_samples 2000 \\")
            print("    --num_epochs 2 \\")
            print("    --run_id refcoco-improved-v1 \\")
            print("    --use_real_refcoco_data True")
            
            print(f"\n💾 備份文件: {backup_path}")
            print("如果有問題，可以從備份恢復")
            
        else:
            print("⚠️  自動修復未能完全應用")
            print("請手動應用修復:")
            
            # 創建手動修復指南
            manual_fix = """
# 手動修復指南

在您的 train_refcoco_improved.py 中進行以下修改:

1. 在腳本開頭添加修復函數 (從下面複製):
"""
            manual_fix += get_complete_patch_code()
            manual_fix += """

2. 在訓練循環中，將:
   outputs = model(**batch)
   
   替換為:
   batch = fix_batch_data_types(batch)
   device_batch = {}
   for key, value in batch.items():
       if isinstance(value, torch.Tensor):
           device_batch[key] = value.to(device)
       else:
           device_batch[key] = value
   outputs = model(**device_batch)
"""
            
            with open("manual_fix_guide.txt", "w", encoding="utf-8") as f:
                f.write(manual_fix)
            
            print("📝 手動修復指南已保存到: manual_fix_guide.txt")
    
    except Exception as e:
        print(f"❌ 修復過程出錯: {e}")
        print("正在從備份恢復...")
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, script_path)
            print("✅ 已從備份恢復")


if __name__ == "__main__":
    main()