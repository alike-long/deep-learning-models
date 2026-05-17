"""
DETR 目标检测 — Transformer 做检测
===================================
思路：
  1. CNN 提取图像特征 (ResNet-18)
  2. 加位置编码
  3. Transformer Encoder → Decoder
  4. 检测头输出 N 个框 + 类别

数据集：自己造玩具数据（白底彩色矩形），自动生成，无需下载
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import math
import random
import numpy as np
from PIL import Image, ImageDraw


# ================================================================
#  第1部分：造玩具数据集（白底 + 随机彩色矩形）
# ================================================================
# 不用下载，运行时会自动生成 200 张小图
# 每张 128×128，白底上画 1~3 个彩色矩形
# 标签: 0=红框, 1=蓝框, 2=绿框
# 框坐标归一化到 [0,1]: cx, cy, w, h

COLORS = ["red", "blue", "green"]


def random_box():
    """随机生成一个归一化框 (cx, cy, w, h) + 类别"""
    w = random.uniform(0.15, 0.4)       # 宽 15%~40%
    h = random.uniform(0.15, 0.4)       # 高 15%~40%
    cx = random.uniform(w / 2, 1 - w / 2)   # 中心不能出界
    cy = random.uniform(h / 2, 1 - h / 2)
    cls = random.randint(0, 2)          # 随机颜色 红/蓝/绿
    return torch.tensor([cx, cy, w, h]), cls


def generate_image(boxes, labels, size=128):
    """根据框列表生成 PIL 图 + 归一化框列表"""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for (cx, cy, w, h), cls in zip(boxes, labels):
        x0 = int((cx - w / 2) * size)
        y0 = int((cy - h / 2) * size)
        x1 = int((cx + w / 2) * size)
        y1 = int((cy + h / 2) * size)
        draw.rectangle([x0, y0, x1, y1], fill=COLORS[cls])
    return img


class ToyDetectionDataset(Dataset):
    """随机生成 200 张检测图，每张 1~3 个矩形"""
    def __init__(self, num=200, max_boxes=5):
        self.samples = []
        for _ in range(num):
            n = random.randint(1, 3)
            boxes, labels = [], []
            for _ in range(n):
                box, cls = random_box()
                boxes.append(box)
                labels.append(cls)
            img = generate_image(boxes, labels)
            # 转 tensor
            img = T.ToTensor()(img)  # (3, 128, 128)
            # pad boxes 到 max_boxes 个，用全0框+label=-1填充
            boxes = torch.stack(boxes)          # (n, 4)
            labels = torch.tensor(labels)       # (n,)
            if n < max_boxes:
                pad_boxes = torch.zeros(max_boxes - n, 4)
                pad_labels = torch.full((max_boxes - n,), -1, dtype=torch.long)
                boxes = torch.cat([boxes, pad_boxes], dim=0)
                labels = torch.cat([labels, pad_labels], dim=0)
            self.samples.append((img, boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ================================================================
#  第2部分：DETR 模型
# ================================================================
class DETR(nn.Module):
    """
    简化版 DETR:
      Backbone (ResNet-18) → 位置编码 → Transformer → 检测头
    """
    def __init__(self, num_classes=3, d_model=128, n_heads=4, n_enc=2, n_dec=2, n_queries=5):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries

        # ---------- ① Backbone: ResNet-18 去掉最后两层 ----------
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)             # 玩具数据不需要预训练
        # 只要前几层，把 3×128×128 变成 128×8×8
        self.backbone = nn.Sequential(
            backbone.conv1,    # 128→64
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # 64→32
            backbone.layer1,   # 32→32
            backbone.layer2,   # 32→16
            backbone.layer3,   # 16→8
        )
        # 最后接 1×1 卷积降维到 d_model
        self.conv_proj = nn.Conv2d(256, d_model, 1)
        #                                       ↑ layer3 输出 256 通道

        # ---------- ② 位置编码（可学习的）----------
        # 特征图 8×8 = 64 个位置
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.1)

        # ---------- ③ Transformer ----------
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_enc)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_dec)

        # ---------- ④ Object Queries（可学习的）----------
        # 5 个 queries → 最多预测 5 个框
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.1)

        # ---------- ⑤ 检测头 ----------
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 是"无物体"类
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),          # 输出归一化到 [0,1]
        )

    def forward(self, x):
        # x: (B, 3, 128, 128)
        B = x.shape[0]

        # ---- 特征提取 ----
        feat = self.backbone(x)              # (B, 256, 8, 8)
        feat = self.conv_proj(feat)          # (B, 128, 8, 8)
        H, W = feat.shape[-2:]               # 8, 8

        # 展平：把 8×8 特征图变成 64 个 token
        feat = feat.flatten(2).transpose(1, 2)  # (B, 64, 128)
        feat = feat + self.pos_embed[:, :64, :] # 加位置编码

        # ---- Transformer Encoder ----
        memory = self.encoder(feat)          # (B, 64, 128)

        # ---- Transformer Decoder ----
        query = self.query_embed.expand(B, -1, -1)  # (B, 5, 128)
        decoded = self.decoder(query, memory)        # (B, 5, 128)

        # ---- 检测头 ----
        class_logits = self.class_head(decoded)      # (B, 5, num_classes+1)
        bboxes = self.bbox_head(decoded)              # (B, 5, 4)

        return class_logits, bboxes


# ================================================================
#  第3部分：损失函数
# ================================================================
def detr_loss(pred_logits, pred_boxes, gt_boxes, gt_labels):
    """
    简化版匹配 + 损失（不用匈牙利算法，因为玩具数据框已按顺序排好）
    pred_logits: (B, 5, 4)  ← 0=红 1=蓝 2=绿 3=无物体
    pred_boxes:  (B, 5, 4)  ← (cx, cy, w, h) 归一化
    gt_boxes:    (B, 5, 4)  ← 已 pad
    gt_labels:   (B, 5)     ← 已 pad, -1=无
    """
    B, N = pred_logits.shape[:2]

    # 分类损失
    # 把 gt_labels 中 -1 改成 3（"无物体"类）
    gt_cls = gt_labels.clone()
    gt_cls[gt_cls < 0] = 3    # -1 → 3（我们的无物体类是第3类）
    loss_cls = F.cross_entropy(pred_logits.reshape(-1, 4), gt_cls.reshape(-1))

    # 框回归损失（只算真实框的位置）
    mask = (gt_labels >= 0).float().unsqueeze(-1)  # (B, 5, 1)
    loss_box = F.l1_loss(pred_boxes * mask, gt_boxes * mask, reduction='sum') / (mask.sum() + 1e-6)

    return loss_cls + 3 * loss_box


# ================================================================
#  第4部分：训练
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ToyDetectionDataset(num=300)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DETR(num_classes=3).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    #                      ↑ DETR 需要小学习率，尤其 Transformer 部分

    print(f"设备: {device} | 样本: {len(dataset)}")
    for epoch in range(30):
        total_loss = 0
        for imgs, boxes, labels in loader:
            # imgs: (16, 3, 128, 128)
            # boxes: (16, 5, 4)  labels: (16, 5)
            imgs = imgs.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            pred_logits, pred_boxes = model(imgs)
            # pred_logits: (16, 5, 4)  pred_boxes: (16, 5, 4)
            loss = detr_loss(pred_logits, pred_boxes, boxes, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss: {total_loss/len(loader):.4f}")

    return model, device


# ================================================================
#  第5部分：推理 + 可视化
# ================================================================
def draw_boxes(img_tensor, boxes, scores, size=128):
    """把预测框画在图上，返回 PIL 图片"""
    img = T.ToPILImage()(img_tensor)
    draw = ImageDraw.Draw(img)

    for j in range(len(boxes)):
        cls_id = scores[j].argmax().item()
        conf = scores[j].max().item()
        if cls_id == 3 or conf < 0.5:      # 无物体 或 置信度低
            continue
        cx, cy, w, h = boxes[j].tolist()
        x0 = int((cx - w / 2) * size)
        y0 = int((cy - h / 2) * size)
        x1 = int((cx + w / 2) * size)
        y1 = int((cy + h / 2) * size)
        # 画框 + 标签
        draw.rectangle([x0, y0, x1, y1], outline=COLORS[cls_id], width=2)
        draw.text((x0 + 2, y0 - 10), COLORS[cls_id], fill=COLORS[cls_id])
    return img


def predict_and_show(model, device):
    model.eval()
    dataset = ToyDetectionDataset(num=4)      # 随机生成 4 张测试图

    print("\n=== 检测结果（图片保存到当前目录）===")
    for i, (img_tensor, gt_boxes, gt_labels) in enumerate(dataset):
        with torch.no_grad():
            logits, boxes = model(img_tensor.unsqueeze(0).to(device))
            scores = F.softmax(logits, dim=-1)[0]   # (5, 4)
            boxes = boxes[0].cpu()                    # (5, 4)

        # 画出检测框
        result_img = draw_boxes(img_tensor, boxes, scores)
        filename = f"detr_result_{i+1}.png"
        result_img.save(filename)

        # 也打印框信息
        n_found = 0
        for j in range(5):
            cls_id = scores[j].argmax().item()
            conf = scores[j].max().item()
            if cls_id == 3 or conf < 0.5:
                continue
            n_found += 1
            cx, cy, w, h = boxes[j].tolist()
            print(f"  图{i+1} 框{j+1}: {COLORS[cls_id]:5s} conf={conf:.2f} "
                  f"框:({cx:.2f},{cy:.2f},{w:.2f},{h:.2f})")
        if n_found == 0:
            print(f"  图{i+1}: 未检测到物体")

    print(f"\n  已保存: detr_result_1~4.png")


# ================================================================
#  第6部分：运行
# ================================================================
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== DETR 训练 ===")
    print("数据集: 自造玩具数据, 128×128, 红蓝绿矩形")
    print("模型: ResNet-18 + Transformer (2层encoder + 2层decoder)")
    print()

    model, device = train()
    predict_and_show(model, device)
