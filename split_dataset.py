#!/usr/bin/env python3
"""
資料集分割腳本
將 disease/ 目錄下已分類的病害資料分割成訓練集和驗證集
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='分割資料集為訓練集和驗證集')

    parser.add_argument('--source-dir', type=str, default='disease',
                        help='來源資料目錄 (預設: disease)')
    parser.add_argument('--target-dir', type=str, default='dataset',
                        help='目標資料目錄 (預設: dataset)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='驗證集比例 (預設: 0.2，即 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')
    parser.add_argument('--copy', action='store_true',
                        help='複製檔案而非移動 (預設: 移動)')

    return parser.parse_args()


def split_dataset(source_dir, target_dir, val_ratio=0.2, seed=42, copy_files=False):
    """
    分割資料集

    Args:
        source_dir: 來源目錄 (例如 disease/)
        target_dir: 目標目錄 (例如 dataset/)
        val_ratio: 驗證集比例
        seed: 隨機種子
        copy_files: True=複製檔案，False=移動檔案
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 檢查來源目錄
    if not source_path.exists():
        print(f"❌ 錯誤：找不到來源目錄 '{source_dir}'")
        return False

    # 建立目標目錄結構
    train_dir = target_path / 'train'
    val_dir = target_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"來源目錄: {source_path.absolute()}")
    print(f"目標目錄: {target_path.absolute()}")
    print(f"驗證集比例: {val_ratio * 100:.1f}%")
    print(f"操作模式: {'複製' if copy_files else '移動'}")
    print(f"隨機種子: {seed}")
    print("=" * 60)

    # 設定隨機種子
    random.seed(seed)

    # 支援的圖片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPG', '.JPEG', '.PNG'}

    # 獲取所有類別目錄
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ 錯誤：在 '{source_dir}' 下沒有找到任何類別目錄")
        return False

    class_dirs.sort()
    total_train = 0
    total_val = 0

    # 處理每個類別
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n處理類別: {class_name}")

        # 獲取該類別的所有圖片
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix in valid_extensions
        ]

        if not image_files:
            print(f"  ⚠️  警告：類別 '{class_name}' 沒有圖片，跳過")
            continue

        num_images = len(image_files)
        print(f"  找到 {num_images} 張圖片")

        # 分割訓練集和驗證集
        if num_images == 1:
            print(f"  ⚠️  警告：只有 1 張圖片，全部放入訓練集")
            train_files = image_files
            val_files = []
        else:
            train_files, val_files = train_test_split(
                image_files,
                test_size=val_ratio,
                random_state=seed
            )

        # 建立類別子目錄
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)

        # 複製/移動訓練集檔案
        for img_file in train_files:
            target_file = train_class_dir / img_file.name
            if copy_files:
                shutil.copy2(img_file, target_file)
            else:
                shutil.move(str(img_file), target_file)

        # 複製/移動驗證集檔案
        for img_file in val_files:
            target_file = val_class_dir / img_file.name
            if copy_files:
                shutil.copy2(img_file, target_file)
            else:
                shutil.move(str(img_file), target_file)

        total_train += len(train_files)
        total_val += len(val_files)

        print(f"  ✓ 訓練集: {len(train_files)} 張")
        print(f"  ✓ 驗證集: {len(val_files)} 張")

    print("\n" + "=" * 60)
    print("分割完成！")
    print("=" * 60)
    print(f"類別總數: {len(class_dirs)}")
    print(f"訓練集總圖片數: {total_train}")
    print(f"驗證集總圖片數: {total_val}")
    print(f"總圖片數: {total_train + total_val}")
    print()
    print(f"資料集已儲存至: {target_path.absolute()}")
    print(f"  ├── train/ ({total_train} 張圖片)")
    print(f"  └── val/   ({total_val} 張圖片)")

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("資料集分割工具")
    print("=" * 60)
    print()

    success = split_dataset(
        args.source_dir,
        args.target_dir,
        args.val_ratio,
        args.seed,
        args.copy
    )

    if success:
        print("\n下一步：執行資料驗證")
        print(f"  python check_data.py --data-dir {args.target_dir}")
    else:
        print("\n❌ 分割失敗，請檢查錯誤訊息")

    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
