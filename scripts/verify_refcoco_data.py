#!/usr/bin/env python3
"""
验证RefCOCO数据集设置
"""
import json
from pathlib import Path
import argparse


def verify_refcoco_data(data_root: str = "data/refcoco", dataset_name: str = "refcoco"):
    """验证RefCOCO数据设置"""
    data_path = Path(data_root)
    
    print(f"Verifying RefCOCO data at: {data_path}")
    
    # 检查可能的标注文件
    possible_files = [
        f"refs({dataset_name}).json",
        f"{dataset_name}.json",
        "refcoco.json", 
        "refs.json"
    ]
    
    annotation_file = None
    for file in possible_files:
        if (data_path / file).exists():
            annotation_file = data_path / file
            break
    
    if annotation_file is None:
        print(f"❌ No annotation file found. Looking for: {possible_files}")
        return False
    else:
        print(f"✅ Found annotation file: {annotation_file}")
    
    # 检查图像目录
    images_dir = data_path / "images"
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # 检查train2014和val2014
    train_dir = images_dir / "train2014"
    val_dir = images_dir / "val2014"
    
    if train_dir.exists():
        train_count = len(list(train_dir.glob("*.jpg")))
        print(f"✅ Found {train_count} training images")
    else:
        print(f"❌ Training images not found: {train_dir}")
    
    if val_dir.exists():
        val_count = len(list(val_dir.glob("*.jpg")))
        print(f"✅ Found {val_count} validation images")
    else:
        print(f"❌ Validation images not found: {val_dir}")
    
    # 检查JSON文件格式
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"✅ Found {len(data)} examples in list format")
            # 检查第一个例子的格式
            if len(data) > 0:
                example = data[0]
                required_fields = ["expression", "bbox"]
                missing = [f for f in required_fields if f not in example]
                if missing:
                    print(f"⚠️  First example missing fields: {missing}")
                else:
                    print("✅ Example format looks good")
        elif isinstance(data, dict):
            if "refs" in data:
                print(f"✅ Found refer format with {len(data['refs'])} refs")
                if "images" in data:
                    print(f"✅ Found {len(data['images'])} image entries")
                if "annotations" in data:
                    print(f"✅ Found {len(data['annotations'])} annotations")
            else:
                print(f"✅ Found dict format with keys: {list(data.keys())}")
        
        print("✅ RefCOCO data verification passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading annotation file: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/refcoco", help="Path to RefCOCO data")
    parser.add_argument("--dataset_name", default="refcoco", help="Dataset name")
    args = parser.parse_args()
    
    verify_refcoco_data(args.data_root, args.dataset_name)