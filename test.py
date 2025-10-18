import torch
import clip
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


# 定义分类头（与训练时相同）
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


def load_model(model_path, device):
    """加载训练好的模型"""
    # 加载CLIP基础模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    model = clip_model.visual

    # 添加分类头
    model.classifier = ClassificationHead(512, 10).to(device)

    # 加载训练权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['visual_state_dict'])
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])

    model.eval()
    return model


def preprocess_image(image_path, device):
    """预处理输入图像"""
    # 图像预处理（与训练时相同）
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 灰度转RGB
        transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    # 加载图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    return image_tensor, image


def predict_digit(model, image_tensor):
    """预测数字"""
    with torch.no_grad():
        # 提取特征
        features = model(image_tensor)
        # 分类预测
        outputs = model.classifier(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 模型路径
    model_path = "clip_mnist.pth"  # 修改为您的模型路径

    # 加载模型
    print("加载模型中...")
    model = load_model(model_path, device)
    print("模型加载完成!")

    while True:
        # 输入图像路径
        image_path = input("\n请输入图像路径（输入'quit'退出）: ")

        if image_path.lower() == 'quit':
            break

        try:
            # 预处理图像
            image_tensor, original_image = preprocess_image(image_path, device)

            # 进行预测
            predicted_digit, confidence, all_probabilities = predict_digit(model, image_tensor)

            # 显示结果
            plt.figure(figsize=(12, 4))

            # 显示原图
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f'Input Image\nPredicted: {predicted_digit} (Confidence: {confidence:.3f})')
            plt.axis('off')

            # 显示概率分布
            plt.subplot(1, 2, 2)
            digits = list(range(10))
            bars = plt.bar(digits, all_probabilities, color='skyblue')
            bars[predicted_digit].set_color('red')  # 最高概率的柱子标红
            plt.xlabel('number')
            plt.ylabel('probability')
            plt.title('probability of each digit')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

            # 在柱子上添加概率值
            for i, prob in enumerate(all_probabilities):
                plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

            print(f"\n预测结果: 数字 {predicted_digit}")
            print(f"置信度: {confidence:.3f}")
            print("\n所有数字的概率分布:")
            for digit, prob in enumerate(all_probabilities):
                print(f"  数字 {digit}: {prob:.3f}")

        except FileNotFoundError:
            print(f"错误: 找不到文件 '{image_path}'")
        except Exception as e:
            print(f"处理图像时出错: {e}")


if __name__ == "__main__":
    main()