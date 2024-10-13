import torch
from torchvision import transforms
from timm import create_model, list_models
from PIL import Image


model_names = list_models(pretrained=True)
# print(f'available models:\n {model_names}')


# # 选择一个预训练模型
# model_name = 'deeplabv3_resnet50'
# pretrained_model = create_model(model_name, pretrained=True)

# # 切换到评估模式，关闭dropout和batch normalization层
# pretrained_model.eval()

# # 定义预处理变换
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 加载图像
# image_path = r"C:\Users\\Desktop\ai_design\00019-2600014894.png"
# image = Image.open(image_path).convert('RGB')

# # 应用预处理变换
# image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

# # 如果有GPU，将图像和数据模型转移到GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_tensor = image_tensor.to(device)
# pretrained_model = pretrained_model.to(device)

# # 提取特征
# with torch.no_grad():  # 不需要计算梯度，节省内存和计算资源
#     features = pretrained_model.forward_features(image_tensor)  # 获取特征

# # 将特征转移到CPU（如果需要）并展平
# features = features.cpu().numpy().flatten()

# print(features)
# print(features.max())
# print(features.min())

#####################################################################################
model = create_model("resnet50", pretrained=True)

# 将模型设置为评估模式
model.eval()

# 加载图像并进行预处理
image = Image.open(r"C:\Users\\Desktop\ai_design\z2.jpg")
# 定义图像预处理转换
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 转换为Tensor类型
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)

# 应用预处理转换
processed_image = preprocess(image)

# 查看处理后的图像形状和数值范围
print("Processed image shape:", processed_image.shape)
print("Processed image range:", processed_image.min(), "-", processed_image.max())

# 5. 可选：将处理后的图像转换回PIL图像对象
processed_pil_image = transforms.ToPILImage()(processed_image)

# 6. 可选：显示处理后的图像
processed_pil_image.show()


# 将输入张量转换为批处理张量
input_batch = processed_image.unsqueeze(0)

# 将输入张量传递给模型并获取输出
with torch.no_grad():
    output = model(input_batch)

# 获取预测结果
_, predicted = torch.max(output.data, 1)

# 打印预测结果
print(predicted)
print(predicted.item())
