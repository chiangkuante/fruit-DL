#!/usr/bin/env python3
"""
植物病蟲害辨識模型訓練程式
使用 ConvNeXt Large 進行遷移學習
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='訓練植物病蟲害辨識模型')

    # 資料路徑
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='資料集根目錄 (預設: dataset)')

    # 超參數
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小 (預設: 8，若 OOM 請改用 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練週期數 (預設: 30)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='學習率 (預設: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='權重衰減 (預設: 0.01)')

    # 模型設定
    parser.add_argument('--model-name', type=str, default='convnext_large.fb_in1k',
                        help='timm 模型名稱 (預設: convnext_large.fb_in1k)')

    # 輸出設定
    parser.add_argument('--output-dir', type=str, default='output',
                        help='輸出目錄 (預設: output)')

    # 其他設定
    parser.add_argument('--num-workers', type=int, default=4,
                        help='資料載入執行緒數 (預設: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')

    return parser.parse_args()


def set_seed(seed):
    """設定隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_transforms(config):
    """
    根據模型配置建立資料轉換
    Args:
        config: timm 模型的資料配置
    Returns:
        train_transform, val_transform
    """
    input_size = config['input_size'][-1]  # 取得 input size (例如 224)
    mean = config['mean']
    std = config['std']

    # 訓練集：包含資料增強
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 驗證集：僅 resize 和 normalize
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def create_dataloaders(data_dir, train_transform, val_transform, batch_size, num_workers):
    """建立訓練和驗證資料載入器"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 建立 ImageFolder 資料集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes


def create_model(model_name, num_classes):
    """建立並初始化模型"""
    print(f"正在載入模型: {model_name}")
    print(f"類別數量: {num_classes}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用混合精度訓練
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 統計
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """驗證模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 統計
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, classes, output_dir, epoch, acc, is_best=False):
    """儲存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': acc,
        'classes': classes
    }

    # 儲存當前 epoch 的模型
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # 如果是最佳模型，額外儲存一份
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 儲存最佳模型: {best_path} (準確率: {acc:.2f}%)")


def save_classes_json(classes, output_dir):
    """儲存類別映射表為 JSON 檔案"""
    classes_dict = {i: class_name for i, class_name in enumerate(classes)}

    json_path = os.path.join(output_dir, 'classes.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ 類別映射表已儲存至: {json_path}")


def main():
    args = parse_args()

    # 設定隨機種子
    set_seed(args.seed)

    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 檢查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU 訓練會非常慢！")

    # 建立臨時模型以獲取資料配置
    print("\n正在準備資料轉換...")
    temp_model = timm.create_model(args.model_name, pretrained=False)
    data_config = resolve_data_config({}, model=temp_model)
    print(f"模型輸入尺寸: {data_config['input_size']}")
    print(f"標準化參數 - Mean: {data_config['mean']}, Std: {data_config['std']}")
    del temp_model

    # 建立資料轉換
    train_transform, val_transform = get_data_transforms(data_config)

    # 建立資料載入器
    print(f"\n正在載入資料集: {args.data_dir}")
    train_loader, val_loader, classes = create_dataloaders(
        args.data_dir,
        train_transform,
        val_transform,
        args.batch_size,
        args.num_workers
    )

    print(f"訓練集大小: {len(train_loader.dataset)}")
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"類別數量: {len(classes)}")
    print(f"類別名稱: {classes}")

    # 儲存類別映射表
    save_classes_json(classes, args.output_dir)

    # 建立模型
    print("\n正在建立模型...")
    model = create_model(args.model_name, len(classes))
    model = model.to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 建立混合精度訓練的 GradScaler
    scaler = torch.amp.GradScaler('cuda')

    # 訓練迴圈
    print(f"\n開始訓練 (共 {args.epochs} 個 epochs)")
    print("=" * 80)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # 驗證
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # 印出統計資訊
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # 儲存檢查點
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(model, classes, args.output_dir, epoch, val_acc, is_best)

        print("=" * 80)

    print(f"\n訓練完成！")
    print(f"最佳驗證準確率: {best_acc:.2f}%")
    print(f"模型已儲存至: {args.output_dir}")


if __name__ == '__main__':
    main()
