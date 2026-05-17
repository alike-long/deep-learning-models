# AI 模型 API 服务
from fastapi import FastAPI, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# --------------------- 模型结构（你的LeNet） ---------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(120, 84), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)

# --------------------- 启动API ---------------------
app = FastAPI()

# 加载模型
model = LeNet()
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location="cpu"))
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,),(0.3530,))
])

# 类别名称
class_names = ['T恤','裤子','套头衫','裙子','外套','凉鞋','衬衫','运动鞋','包','靴子']

# --------------------- 接口 ---------------------
@app.post("/predict")
async def predict(file: UploadFile):
    # 读取图片
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    img = transform(image).unsqueeze(0)

    # 预测
    with torch.no_grad():
        output = model(img)
        class_id = torch.max(output, 1)[1].item()

    return {
        "结果": class_names[class_id],
        "编号": class_id
    }

