#!/usr/bin/env python3
"""
資料結構驗證腳本
檢查 dataset/ 目錄是否符合 ImageFolder 格式，並統計類別與圖片數量
"""

import os
from pathlib import Path
from collections import defaultdict


def check_dataset_structure(dataset_path='dataset'):
    """檢查資料集結構是否正確"""
    dataset_dir = Path(dataset_path)

    # 檢查 dataset 目錄是否存在
    if not dataset_dir.exists():
        print(f"❌ 錯誤：找不到 '{dataset_path}' 目錄")
        print(f"   請確保資料集位於: {dataset_dir.absolute()}")
        return False

    print(f"✓ 找到資料集目錄: {dataset_dir.absolute()}")
    print()

    # 檢查 train 和 val 子目錄
    required_splits = ['train', 'val']
    missing_splits = []

    for split in required_splits:
        split_path = dataset_dir / split
        if not split_path.exists():
            missing_splits.append(split)

    if missing_splits:
        print(f"❌ 錯誤：缺少必要的子目錄: {', '.join(missing_splits)}")
        print(f"   資料集應包含以下結構:")
        print(f"   {dataset_path}/")
        print(f"     ├── train/")
        print(f"     └── val/")
        return False

    # 統計各分割的類別和圖片數量
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    all_valid = True

    for split in required_splits:
        split_path = dataset_dir / split
        print(f"{'='*60}")
        print(f"檢查 {split.upper()} 資料集")
        print(f"{'='*60}")

        # 獲取所有類別目錄
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]

        if not class_dirs:
            print(f"❌ 錯誤：{split}/ 目錄下沒有找到任何類別子目錄")
            all_valid = False
            continue

        class_dirs.sort()
        total_images = 0
        class_stats = []

        for class_dir in class_dirs:
            class_name = class_dir.name

            # 統計圖片數量
            image_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            num_images = len(image_files)
            total_images += num_images

            class_stats.append((class_name, num_images))

            if num_images == 0:
                print(f"  ⚠️  警告：類別 '{class_name}' 沒有圖片")

        print(f"\n類別總數: {len(class_dirs)}")
        print(f"圖片總數: {total_images}")
        print(f"\n各類別圖片數量:")
        print(f"{'類別名稱':<30} {'圖片數量':>10}")
        print(f"{'-'*42}")

        for class_name, num_images in class_stats:
            status = "✓" if num_images > 0 else "⚠"
            print(f"{status} {class_name:<28} {num_images:>10}")

        print()

    # 驗證 train 和 val 的類別是否一致
    train_classes = set(d.name for d in (dataset_dir / 'train').iterdir() if d.is_dir())
    val_classes = set(d.name for d in (dataset_dir / 'val').iterdir() if d.is_dir())

    if train_classes != val_classes:
        print(f"{'='*60}")
        print("⚠️  警告：訓練集和驗證集的類別不一致")
        print(f"{'='*60}")

        only_in_train = train_classes - val_classes
        only_in_val = val_classes - train_classes

        if only_in_train:
            print(f"僅在訓練集: {', '.join(sorted(only_in_train))}")
        if only_in_val:
            print(f"僅在驗證集: {', '.join(sorted(only_in_val))}")
        print()
    else:
        print(f"{'='*60}")
        print("✓ 訓練集和驗證集類別一致")
        print(f"{'='*60}")
        print()

    if all_valid:
        print("✓ 資料集結構驗證完成！可以開始訓練。")
        return True
    else:
        print("❌ 資料集結構有問題，請修正後再執行訓練。")
        return False


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='驗證資料集結構')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='資料集目錄 (預設: dataset)')
    args = parser.parse_args()

    print("=" * 60)
    print("植物病蟲害資料集結構驗證")
    print("=" * 60)
    print()

    success = check_dataset_structure(args.data_dir)
    sys.exit(0 if success else 1)
