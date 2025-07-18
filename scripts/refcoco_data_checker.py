#!/usr/bin/env python3
"""
refcoco_data_checker.py

檢查和修復RefCOCO數據集的工具
解決數據類型和路徑問題
"""
import json
import zipfile
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


def extract_zip_file(zip_path: Path, extract_to: Path) -> bool:
    """解壓ZIP檔案"""
    try:
        print(f"Extracting {zip_path} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Successfully extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False


def find_json_files(directory: Path) -> List[Path]:
    """在目錄中尋找JSON檔案"""
    json_files = []
    for file_path in directory.rglob("*.json"):
        json_files.append(file_path)
    return json_files


def analyze_json_structure(json_path: Path) -> Dict[str, Any]:
    """分析JSON檔案結構"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            "file_path": str(json_path),
            "file_size_mb": json_path.stat().st_size / (1024 * 1024),
            "data_type": type(data).__name__,
            "structure": {}
        }
        
        if isinstance(data, dict):
            analysis["structure"]["keys"] = list(data.keys())
            analysis["structure"]["key_types"] = {k: type(v).__name__ for k, v in data.items()}
            
            # 分析特定鍵的內容
            for key in ["annotations", "images", "refs", "categories"]:
                if key in data:
                    if isinstance(data[key], list) and len(data[key]) > 0:
                        sample = data[key][0]
                        analysis["structure"][f"{key}_sample"] = {
                            "count": len(data[key]),
                            "sample_keys": list(sample.keys()) if isinstance(sample, dict) else "not_dict",
                            "sample_types": {k: type(v).__name__ for k, v in sample.items()} if isinstance(sample, dict) else {}
                        }
        
        elif isinstance(data, list):
            analysis["structure"]["list_length"] = len(data)
            if len(data) > 0:
                sample = data[0]
                analysis["structure"]["sample"] = {
                    "sample_keys": list(sample.keys()) if isinstance(sample, dict) else "not_dict",
                    "sample_types": {k: type(v).__name__ for k, v in sample.items()} if isinstance(sample, dict) else {}
                }
        
        return analysis
    
    except Exception as e:
        return {"error": str(e), "file_path": str(json_path)}


def check_bbox_format(data: Any) -> Dict[str, Any]:
    """檢查邊界框格式"""
    bbox_issues = {
        "total_checked": 0,
        "string_bboxes": 0,
        "invalid_bboxes": 0,
        "valid_bboxes": 0,
        "bbox_samples": []
    }
    
    def check_single_bbox(bbox, context=""):
        bbox_issues["total_checked"] += 1
        
        if isinstance(bbox, str):
            bbox_issues["string_bboxes"] += 1
            bbox_issues["bbox_samples"].append({
                "type": "string",
                "value": bbox,
                "context": context
            })
        elif isinstance(bbox, (list, tuple)):
            if len(bbox) != 4:
                bbox_issues["invalid_bboxes"] += 1
                bbox_issues["bbox_samples"].append({
                    "type": "invalid_length",
                    "value": bbox,
                    "context": context
                })
            else:
                # 檢查是否所有元素都是數字
                try:
                    [float(x) for x in bbox]
                    bbox_issues["valid_bboxes"] += 1
                except (ValueError, TypeError):
                    bbox_issues["invalid_bboxes"] += 1
                    bbox_issues["bbox_samples"].append({
                        "type": "non_numeric",
                        "value": bbox,
                        "context": context
                    })
        else:
            bbox_issues["invalid_bboxes"] += 1
            bbox_issues["bbox_samples"].append({
                "type": "unknown",
                "value": str(bbox),
                "context": context
            })
    
    # 遞歸檢查數據中的bbox
    def recursive_check(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if key in ["bbox", "bounding_box", "box"]:
                    check_single_bbox(value, new_path)
                else:
                    recursive_check(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:10]):  # 只檢查前10個
                recursive_check(item, f"{path}[{i}]")
    
    recursive_check(data)
    return bbox_issues


def fix_data_types(data: Any) -> Any:
    """修復數據類型問題"""
    if isinstance(data, dict):
        fixed = {}
        for key, value in data.items():
            if key in ["bbox", "bounding_box", "box"] and isinstance(value, str):
                # 嘗試解析字符串bbox
                try:
                    # 移除括號和空格，然後分割
                    cleaned = value.strip('[]()').replace(' ', '')
                    bbox_values = [float(x) for x in cleaned.split(',')]
                    if len(bbox_values) == 4:
                        fixed[key] = bbox_values
                    else:
                        print(f"Warning: Invalid bbox format: {value}")
                        fixed[key] = [0.0, 0.0, 1.0, 1.0]
                except:
                    print(f"Warning: Cannot parse bbox: {value}")
                    fixed[key] = [0.0, 0.0, 1.0, 1.0]
            elif key in ["image_id", "category_id", "id"] and isinstance(value, str):
                # 嘗試轉換ID為整數
                try:
                    fixed[key] = int(value)
                except ValueError:
                    fixed[key] = 0
            elif key in ["area", "width", "height"] and isinstance(value, str):
                # 嘗試轉換尺寸為浮點數
                try:
                    fixed[key] = float(value)
                except ValueError:
                    fixed[key] = 0.0
            else:
                fixed[key] = fix_data_types(value)
        return fixed
    elif isinstance(data, list):
        return [fix_data_types(item) for item in data]
    else:
        return data


def check_refcoco_dataset(data_root: Path) -> Dict[str, Any]:
    """全面檢查RefCOCO數據集"""
    print(f"=== Checking RefCOCO Dataset in {data_root} ===")
    
    report = {
        "data_root": str(data_root),
        "exists": data_root.exists(),
        "zip_files": [],
        "json_files": [],
        "image_directories": [],
        "bbox_analysis": {},
        "recommendations": []
    }
    
    if not data_root.exists():
        report["recommendations"].append(f"Create directory: {data_root}")
        return report
    
    # 檢查ZIP檔案
    for zip_file in data_root.rglob("*.zip"):
        report["zip_files"].append({
            "path": str(zip_file),
            "size_mb": zip_file.stat().st_size / (1024 * 1024)
        })
    
    # 檢查JSON檔案
    json_files = find_json_files(data_root)
    for json_file in json_files:
        analysis = analyze_json_structure(json_file)
        report["json_files"].append(analysis)
        
        # 如果是主要的數據文件，進行bbox分析
        if any(keyword in json_file.name.lower() for keyword in ["train", "instance", "refcoco"]):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                bbox_analysis = check_bbox_format(data)
                report["bbox_analysis"][str(json_file)] = bbox_analysis
            except Exception as e:
                report["bbox_analysis"][str(json_file)] = {"error": str(e)}
    
    # 檢查圖片目錄
    for img_dir in data_root.rglob("*"):
        if img_dir.is_dir() and any(keyword in img_dir.name.lower() for keyword in ["image", "img", "train2014", "val2014"]):
            img_count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            report["image_directories"].append({
                "path": str(img_dir),
                "image_count": img_count
            })
    
    # 生成建議
    if report["zip_files"]:
        report["recommendations"].append("Extract ZIP files to access JSON data")
    
    if not report["json_files"]:
        report["recommendations"].append("No JSON files found - check if ZIP files need extraction")
    
    for json_analysis in report["json_files"]:
        if "error" in json_analysis:
            report["recommendations"].append(f"Fix JSON file: {json_analysis['file_path']}")
    
    for file_path, bbox_analysis in report["bbox_analysis"].items():
        if bbox_analysis.get("string_bboxes", 0) > 0:
            report["recommendations"].append(f"Fix string bboxes in {file_path}")
        if bbox_analysis.get("invalid_bboxes", 0) > 0:
            report["recommendations"].append(f"Fix invalid bboxes in {file_path}")
    
    return report


def extract_and_fix_refcoco(data_root: Path, backup: bool = True):
    """解壓並修復RefCOCO數據"""
    print(f"=== Extracting and Fixing RefCOCO Data ===")
    
    # 檢查當前狀態
    report = check_refcoco_dataset(data_root)
    
    # 解壓ZIP檔案
    for zip_info in report["zip_files"]:
        zip_path = Path(zip_info["path"])
        
        # 決定解壓目標
        if "train" in zip_path.name.lower():
            extract_dir = data_root / "extracted_train"
        elif "val" in zip_path.name.lower():
            extract_dir = data_root / "extracted_val"
        else:
            extract_dir = data_root / "extracted"
        
        extract_zip_file(zip_path, extract_dir)
    
    # 尋找並修復JSON檔案
    json_files = find_json_files(data_root)
    
    for json_file in json_files:
        print(f"Processing {json_file}")
        
        try:
            # 備份原檔案
            if backup:
                backup_path = json_file.with_suffix('.json.backup')
                if not backup_path.exists():
                    shutil.copy2(json_file, backup_path)
                    print(f"Backed up to {backup_path}")
            
            # 載入和修復數據
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Original data type: {type(data)}")
            
            # 修復數據類型
            fixed_data = fix_data_types(data)
            
            # 寫回修復的數據
            fixed_path = json_file.with_stem(json_file.stem + "_fixed")
            with open(fixed_path, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, indent=2)
            
            print(f"✅ Fixed data saved to {fixed_path}")
            
            # 驗證修復結果
            bbox_analysis = check_bbox_format(fixed_data)
            print(f"Bbox analysis after fix: valid={bbox_analysis['valid_bboxes']}, "
                  f"invalid={bbox_analysis['invalid_bboxes']}, "
                  f"string={bbox_analysis['string_bboxes']}")
            
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")


def create_training_ready_dataset(data_root: Path, output_path: Path):
    """創建訓練就緒的數據集"""
    print(f"=== Creating Training-Ready Dataset ===")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 尋找最佳的數據文件
    json_files = find_json_files(data_root)
    
    best_file = None
    for json_file in json_files:
        if "fixed" in json_file.name and "train" in json_file.name.lower():
            best_file = json_file
            break
    
    if not best_file:
        for json_file in json_files:
            if "train" in json_file.name.lower() or "instance" in json_file.name.lower():
                best_file = json_file
                break
    
    if not best_file:
        print("❌ No suitable training data file found")
        return
    
    print(f"Using data file: {best_file}")
    
    try:
        with open(best_file, 'r') as f:
            data = json.load(f)
        
        # 轉換為統一格式
        unified_examples = []
        
        if isinstance(data, dict) and "annotations" in data:
            # COCO格式
            images = {img["id"]: img for img in data.get("images", [])}
            
            for ann in data["annotations"][:1000]:  # 限制數量用於測試
                if ann["image_id"] in images:
                    image_info = images[ann["image_id"]]
                    
                    example = {
                        "image_id": str(ann["image_id"]),
                        "image_path": image_info.get("file_name", ""),
                        "expression": ann.get("expression", ann.get("caption", "")),
                        "bbox": ann.get("bbox", [0, 0, 1, 1]),
                        "category_id": ann.get("category_id", 0),
                        "split": "train"
                    }
                    
                    # 確保bbox是正確格式
                    bbox = example["bbox"]
                    if isinstance(bbox, str):
                        try:
                            bbox = [float(x) for x in bbox.strip('[]()').split(',')]
                        except:
                            bbox = [0.0, 0.0, 1.0, 1.0]
                    
                    if len(bbox) == 4:
                        example["bbox"] = [float(x) for x in bbox]
                        unified_examples.append(example)
        
        elif isinstance(data, list):
            # 直接是examples列表
            for item in data[:1000]:
                if isinstance(item, dict):
                    unified_examples.append(item)
        
        # 保存統一格式的數據
        output_file = output_path / "train_ready.json"
        with open(output_file, 'w') as f:
            json.dump(unified_examples, f, indent=2)
        
        print(f"✅ Created training-ready dataset with {len(unified_examples)} examples")
        print(f"Saved to: {output_file}")
        
        # 創建簡化的配置文件
        config = {
            "dataset_name": "refcoco_fixed",
            "data_file": "train_ready.json",
            "images_dir": "images",
            "total_examples": len(unified_examples),
            "bbox_format": "xyxy",
            "description": "Fixed RefCOCO dataset ready for training"
        }
        
        config_file = output_path / "dataset_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to: {config_file}")
        
    except Exception as e:
        print(f"❌ Error creating training dataset: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="RefCOCO Data Checker and Fixer")
    parser.add_argument("--data_root", type=str, default="data", help="Path to RefCOCO data directory")
    parser.add_argument("--action", type=str, choices=["check", "extract", "fix", "prepare"], 
                       default="check", help="Action to perform")
    parser.add_argument("--output", type=str, help="Output directory for prepared dataset")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup original files")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if args.action == "check":
        report = check_refcoco_dataset(data_root)
        print(json.dumps(report, indent=2))
    
    elif args.action == "extract":
        extract_and_fix_refcoco(data_root, backup=not args.no_backup)
    
    elif args.action == "fix":
        extract_and_fix_refcoco(data_root, backup=not args.no_backup)
    
    elif args.action == "prepare":
        output_path = Path(args.output) if args.output else data_root / "training_ready"
        create_training_ready_dataset(data_root, output_path)


if __name__ == "__main__":
    main()