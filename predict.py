#!/usr/bin/env python3
"""
植物病蟲害辨識 - 推論腳本
使用訓練好的模型對單張或多張圖片進行預測
支援命令列使用和程式化呼叫 (適用於 Streamlit 整合)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config


class PlantDiseasePredictor:
    """植物病蟲害預測器類別 - 可重複使用於不同介面"""

    def __init__(
        self,
        model_path: str = 'output/best_model.pth',
        classes_path: str = 'output/classes.json',
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化預測器

        Args:
            model_path: 模型權重檔案路徑
            classes_path: 類別映射 JSON 檔案路徑
            device: 指定裝置 ('cuda', 'cpu', 或 None 自動選擇)
            verbose: 是否顯示載入訊息
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.verbose = verbose

        # 載入類別映射
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes_dict = json.load(f)

        self.num_classes = len(self.classes_dict)
        self.class_names = list(self.classes_dict.values())

        # 載入檢查點
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model_accuracy = checkpoint.get('accuracy', None)

        # 建立模型
        self.model = timm.create_model(
            'convnext_large.fb_in1k',
            pretrained=False,
            num_classes=self.num_classes
        )

        # 載入權重
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 選擇裝置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 建立圖片轉換
        self.transform = self._create_transform()

        if self.verbose:
            self._print_info()

    def _create_transform(self):
        """建立圖片轉換"""
        temp_model = timm.create_model('convnext_large.fb_in1k', pretrained=False)
        data_config = resolve_data_config({}, model=temp_model)

        input_size = data_config['input_size'][-1]
        mean = data_config['mean']
        std = data_config['std']

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    def _print_info(self):
        """顯示模型資訊"""
        print(f"類別數量: {self.num_classes}")
        print(f"類別名稱: {self.class_names}")
        print(f"模型已載入: {self.model_path}")
        print(f"使用裝置: {self.device}")
        if self.model_accuracy:
            print(f"模型準確率: {self.model_accuracy:.2f}%")

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        對單張圖片進行預測

        Args:
            image: 圖片路徑或 PIL Image 物件
            top_k: 回傳前 k 個預測結果

        Returns:
            predictions: [(class_name, probability_percentage), ...]
        """
        # 載入圖片 (如果是路徑)
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise TypeError("image 必須是檔案路徑或 PIL Image 物件")

        # 轉換圖片
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # 獲取 top-k 預測
        top_probs, top_indices = torch.topk(
            probabilities[0],
            min(top_k, self.num_classes)
        )

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.classes_dict[str(idx.item())]
            predictions.append((class_name, prob.item() * 100))

        return predictions

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        top_k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """
        批次預測多張圖片

        Args:
            images: 圖片路徑或 PIL Image 物件列表
            top_k: 回傳前 k 個預測結果

        Returns:
            results: 每張圖片的預測結果列表
        """
        results = []
        for image in images:
            try:
                predictions = self.predict(image, top_k)
                results.append(predictions)
            except Exception as e:
                if self.verbose:
                    print(f"預測失敗: {image} - {e}")
                results.append([])
        return results

    def get_class_names(self) -> List[str]:
        """取得所有類別名稱"""
        return self.class_names

    def get_model_info(self) -> dict:
        """取得模型資訊"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': str(self.device),
            'accuracy': self.model_accuracy
        }


# 保留舊的函數以維持向後相容性
def load_model_and_classes(model_path, classes_path):
    """
    載入模型和類別映射 (向後相容函數)

    Args:
        model_path: 模型權重檔案路徑
        classes_path: 類別映射 JSON 檔案路徑

    Returns:
        model, classes_dict, device
    """
    predictor = PlantDiseasePredictor(model_path, classes_path, verbose=True)
    return predictor.model, predictor.classes_dict, predictor.device


def get_transform():
    """建立圖片轉換 (向後相容函數)"""
    predictor = PlantDiseasePredictor(verbose=False)
    return predictor.transform


def predict_image(image_path, model, classes_dict, transform, device, top_k=3):
    """
    對單張圖片進行預測 (向後相容函數)

    Args:
        image_path: 圖片路徑
        model: 模型
        classes_dict: 類別映射字典
        transform: 圖片轉換
        device: 計算裝置
        top_k: 顯示前 k 個預測結果

    Returns:
        predictions: [(class_name, probability), ...]
    """
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(classes_dict)))

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = classes_dict[str(idx.item())]
        predictions.append((class_name, prob.item() * 100))

    return predictions


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='植物病蟲害辨識推論')

    parser.add_argument('--image', type=str, required=True,
                        help='圖片路徑或包含圖片的目錄')
    parser.add_argument('--model', type=str, default='output/best_model.pth',
                        help='模型權重檔案 (預設: output/best_model.pth)')
    parser.add_argument('--classes', type=str, default='output/classes.json',
                        help='類別映射檔案 (預設: output/classes.json)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='顯示前 k 個預測結果 (預設: 3)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("植物病蟲害辨識 - 推論系統")
    print("=" * 80)
    print()

    # 載入模型
    model, classes_dict, device = load_model_and_classes(args.model, args.classes)
    print()

    # 建立轉換
    transform = get_transform()

    # 獲取圖片路徑列表
    image_path = Path(args.image)
    if image_path.is_dir():
        # 如果是目錄，獲取所有圖片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = [
            f for f in image_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        print(f"找到 {len(image_files)} 張圖片")
        print()
    else:
        # 單張圖片
        image_files = [image_path]

    # 對每張圖片進行預測
    for img_file in image_files:
        print(f"{'='*80}")
        print(f"圖片: {img_file.name}")
        print(f"{'='*80}")

        try:
            predictions = predict_image(
                img_file, model, classes_dict, transform, device, args.top_k
            )

            # 顯示預測結果
            print(f"\n預測結果 (Top {len(predictions)}):")
            print(f"{'排名':<6} {'類別':<20} {'信心度':>10}")
            print("-" * 40)

            for rank, (class_name, prob) in enumerate(predictions, 1):
                confidence_bar = "█" * int(prob / 5)  # 20個字符代表100%
                print(f"{rank:<6} {class_name:<20} {prob:>9.2f}% {confidence_bar}")

            # 標註最可能的預測
            best_class, best_prob = predictions[0]
            print(f"\n✓ 最可能的診斷: {best_class} (信心度: {best_prob:.2f}%)")

        except Exception as e:
            print(f"❌ 處理圖片時發生錯誤: {e}")

        print()


if __name__ == '__main__':
    main()
