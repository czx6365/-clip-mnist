import torch
import clip  # 需要安装OpenAI的CLIP包: pip install git+https://github.com/openai/CLIP.git
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def convert_to_rgb(grayscale_img):
    a = grayscale_img.repeat(3, 1, 1)
    return a


# 1. 数据加载（需调整预处理适配CLIP）
clip_preprocess = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    convert_to_rgb,
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))  # CLIP的标准化参数
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=clip_preprocess)

test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=clip_preprocess)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

# 2. 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# 3. 原CLIP输出512维，MNIST需要10类
model = model.visual  # 只取图像编码器部分
model.output_dim = 512


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # 二分类用sigmoid


model.classifier = ClassificationHead(model.output_dim, 10).to(device)

# 4. 损失函数和优化器（MNIST是多分类，应使用CrossEntropy）
criterion = torch.nn.CrossEntropyLoss()  # 替换原来的BCELoss
optimizer = torch.optim.Adam([
    {'params': model.parameters()},

], lr=1e-4)

# 5. 训练循环（需调整预测处理）
epochs = 3
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 提取CLIP图像特征
        with torch.no_grad():  # 冻结CLIP主干（可选）
            features = model(images)

        # 仅训练分类头
        outputs = model.classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch: {epoch} Loss: {loss.item():.4f}')

# 6. 评估（修改预测处理）
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        features = model(images)
        outputs = model.classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'Accuracy: {100 * correct / total:.2f}%')

# 7. 保存/加载模型（需特殊处理CLIP结构）
torch.save({
    'visual_state_dict': model.state_dict(),
    'classifier_state_dict': model.classifier.state_dict()
}, 'clip_mnist.pth')

# 加载时需先初始化模型再加载权重
loaded_model, _ = clip.load("ViT-B/32", device=device)
loaded_model = loaded_model.visual
loaded_model.classifier = ClassificationHead(512, 10).to(device)
checkpoint = torch.load('clip_mnist.pth')
loaded_model.load_state_dict(checkpoint['visual_state_dict'])
loaded_model.classifier.load_state_dict(checkpoint['classifier_state_dict'])

# 8. 单图预测示例
image, label = test_data[0]
image = image.unsqueeze(0).to(device)
with torch.no_grad():
    feature = loaded_model(image)
    output = loaded_model.classifier(feature)
    _, predicted = torch.max(output, 1)

plt.imshow(image.squeeze().cpu().permute(1, 2, 0)[:, :, 0])  # 显示灰度通道
plt.title(f'Predicted: {predicted.item()}, True: {label}')
plt.show()