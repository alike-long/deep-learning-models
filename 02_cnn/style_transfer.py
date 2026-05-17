"""
from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO("yolov8n.pt")

# 你的图片路径（改成你自己的）
img_path = "2290.jpg_wh860.jpg"

# 读取图片
img = cv2.imread(img_path)
h, w = img.shape[:2]

# ===================== 图片偏移（你要的功能） =====================
tx = 0   # 向右偏移30像素
ty = 0   # 向下偏移20像素
M = np.float32([[1,0,tx], [0,1,ty]])
img = cv2.warpAffine(img, M, (w, h))  # 偏移后的图

# 检测人（类别0 = 人）
results = model(img, classes=[0])

# 画框
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 保存结果
cv2.imwrite("rsult.jpg", img)
print("✅ 检测完成！已保存 result.jpg")
"""
"""
class Model (nn.Module) :
    def __init__(self,output) :
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.classifier = nn.Sequential(nn.Linear(64 * 7 * 7, 1024))
    def forward(self, x):
        x = self.features(x)
        x=torch.flatten(x,1)
        x = self.classifier(x)
        return x
"""
"""
import torch
import torchvision
import matplotlib.pyplot as plt
import d2l.torch as d2l
from ultralytics import YOLO

# ===================== 设备 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 直接调用 YOLOv8 官方模型 =====================
model = YOLO('yolov8n.pt')  # 最轻量最快
model.to(device)

# ===================== 你的图片读取（完全不变） =====================
X = torchvision.io.read_image('02d9a1bba3933a3ffacca2e960ba4bd862d9fe25.jpg').unsqueeze(0).float() / 255.0
img = X.squeeze(0).permute(1, 2, 0).cpu().numpy()


# ===================== 你的 predict 函数（风格一样） =====================
def predict(X):
    # YOLOv8 推理
    results = model(X.to(device), verbose=False)
    return results[0]


# ===================== 你的 display 画图（完全一致） =====================
def display(img, output, threshold=0.5):
    d2l.set_figsize((5, 5))
    fig = plt.imshow(img)

    # 提取框、置信度
    boxes = output.boxes.xyxy
    scores = output.boxes.conf

    for box, score in zip(boxes, scores):
        score = float(score)
        if score < threshold:
            continue
        # 画框（和你原来代码一模一样）
        d2l.show_bboxes(fig.axes, [box.cpu()], f'{score:.2f}', 'w')

    plt.show()


# ===================== 运行 =====================
output = predict(X)
display(img, output, threshold=0.5)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

# ===================== 配置路径（自己改这里）=====================
CONTENT_IMG_PATH = r"F:\py-code\vection-work\photo\mmexport1756287856301.jpg"  # 内容图路径
STYLE_IMG_PATH   = r"F:\py-code\vection-work\photo\c8930d77c3ef4a5d889b7c55a9dc29ef~tplv-a9rns2rl98-pc_smart_face_crop-v1_512_384.jpg"    # 风格图路径
OUTPUT_IMG_PATH  = "result.jpg"   # 输出图路径

# 超参数
EPOCHS = 450
LR = 0.001

# ===================== 工具函数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, imsize=512):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)

def show_img(tensor, title=None):
    img = tensor.cpu().clone().squeeze(0)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

# ===================== 加载图像 =====================
content_img = load_image(CONTENT_IMG_PATH)
style_img   = load_image(STYLE_IMG_PATH)
input_img   = content_img.clone()  # 从内容图初始化

# ===================== 损失函数 =====================
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = self.gram(target).detach()
    def gram(self, x):
        a, b, c, d = x.size()
        feat = x.view(a*b, c*d)
        G = torch.mm(feat, feat.t())
        return G / (a*b*c*d)
    def forward(self, x):
        g = self.gram(x)
        self.loss = nn.functional.mse_loss(g, self.target)
        return x

# ===================== 构建模型 =====================
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# 提取层
content_layers = ['conv_4']
style_layers   = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def build_model_and_loss(cnn, content_img, style_img):
    model = nn.Sequential()
    content_losses = []
    style_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            raise RuntimeError("不识别层")
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img)
            cl = ContentLoss(target)
            model.add_module(f'content_loss_{i}', cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img)
            sl = StyleLoss(target)
            model.add_module(f'style_loss_{i}', sl)
            style_losses.append(sl)

    # 裁剪到最后一个损失层
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]
    return model, style_losses, content_losses

model, style_losses, content_losses = build_model_and_loss(cnn, content_img, style_img)

# ===================== 优化 =====================
optimizer = optim.LBFGS([input_img.requires_grad_()])

print("开始风格迁移...")
run = [0]
while run[0] < EPOCHS:
    def closure():
        input_img.data.clamp_(0,1)
        optimizer.zero_grad()
        model(input_img)

        s_loss = sum(sl.loss for sl in style_losses) * 1e4
        c_loss = sum(cl.loss for cl in content_losses)
        loss = s_loss + c_loss
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"step {run[0]} | style: {s_loss.item():.2f} | content: {c_loss.item():.2f}")
        return loss
    optimizer.step(closure)

input_img.data.clamp_(0,1)

# 保存结果
img = input_img.cpu().clone().squeeze(0)
img = transforms.ToPILImage()(img)
img.save(OUTPUT_IMG_PATH)
print(f"迁移完成！已保存到 {OUTPUT_IMG_PATH}")

# 显示
plt.figure()
show_img(input_img, title='Result')


