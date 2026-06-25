"""
YOLO 目标检测 — 简化版
=======================
单阶段检测：图片进 → CNN → 直接输出框+类别

经典 YOLO 三个核心：
  ① 网格划分：图片分成 S×S 格，每格负责预测中心落在里面的物体
  ② 单阶段预测：不用"先找候选区域再分类"，一次性出框
  ③ NMS 后处理：去掉重叠的冗余框，只保留最准的那个

数据集：自造玩具数据 — 白底彩色矩形，类似 DETR 的ToyDetectionDataset

[模型结构]
  CNN Backbone → 两个检测头:
    box_head:  每格预测 (cx,cy,w,h)
    class_head: 每格预测 num_classes
"""
import torch, torch.nn as nn, torch.optim as optim, random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.manual_seed(42)


# ================================================================
#  1. 自造检测数据集（玩具数据）
# ================================================================
COLORS = [(255,0,0), (0,0,255), (0,255,0)]              # 红/蓝/绿
CLASS_NAMES = ['red', 'blue', 'green']
GRID_S = 7                                                # 7×7 网格

class ToyDetectionDataset(Dataset):
    """每张 224×224，白底 1~3 个彩色矩形，S=7网格 — 每格最多1个框"""
    def __init__(self, n=500):
        self.samples = []
        for _ in range(n):
            img = Image.new('RGB', (224, 224), 'white')
            draw = ImageDraw.Draw(img)

            # 每个网格的标签: [cx, cy, w, h, conf, class_id]
            label = torch.zeros(GRID_S, GRID_S, 6)
            label[:, :, 4] = 1.0                             # 默认 conf=1 (置信度; 0=无物体)

            n_objs = random.randint(1, 3)
            for _ in range(n_objs):
                cls_id = random.randint(0, len(COLORS)-1)
                w, h = random.uniform(0.08, 0.25), random.uniform(0.08, 0.25)
                cx, cy = random.uniform(w/2, 1-w/2), random.uniform(h/2, 1-h/2)

                # 画矩形
                x0 = int((cx-w/2) * 224); y0 = int((cy-h/2) * 224)
                x1 = int((cx+w/2) * 224); y1 = int((cy+h/2) * 224)
                draw.rectangle([x0, y0, x1, y1], fill=COLORS[cls_id])

                # 确定中心落在哪个网格
                gx = min(int(cx * GRID_S), GRID_S-1)
                gy = min(int(cy * GRID_S), GRID_S-1)

                # 相对该网格的偏移和尺寸
                cx_cell = cx * GRID_S - gx
                cy_cell = cy * GRID_S - gy
                w_cell = w * GRID_S
                h_cell = h * GRID_S

                label[gy, gx] = torch.tensor([cx_cell, cy_cell, w_cell, h_cell, 0.0, cls_id])

            img_tensor = T.ToTensor()(img)                   # (3, 224, 224)
            self.samples.append((img_tensor, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ================================================================
#  2. YOLO 模型
# ================================================================
class SimpleYOLO(nn.Module):
    """
    Backbone: 3层 Conv + Pool → (B, 128, 7, 7)
    检测头:  两个 1×1 Conv → 框(4) + 有无物体(1) + 类别(3)
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone (小CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 224→112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 112→56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 56→28
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 28→14
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 14→7 ← 对齐网格
        )
        # 检测头 (1×1 卷积 → 逐点预测)
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 5, 1),                             # (cx,cy,w,h,conf)
            nn.Sigmoid(),                                    # 归一化到 [0,1]
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),                   # 每类得分
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.backbone(x)                              # (B, 128, 7, 7)
        boxes = self.box_head(feat)                          # (B, 5, 7, 7)
        #   → (B, 5, 7, 7) → (B, 7, 7, 5)
        boxes = boxes.permute(0, 2, 3, 1)                   # (B, 7, 7, 5)
        classes = self.cls_head(feat).permute(0, 2, 3, 1)  # (B, 7, 7, C)
        return boxes, classes


# ================================================================
#  3. 损失函数
# ================================================================
def yolo_loss(pred_boxes, pred_classes, targets):
    """
    pred_boxes:  (B,7,7,5)  [cx,cy,w,h,conf]
    pred_classes:(B,7,7,C)
    targets:     (B,7,7,6)  [cx,cy,w,h,conf,cls]
    """

    B = pred_boxes.shape[0]

    # ★ 哪些格有物体？(conf = 0 = 有物体)
    obj_mask = (targets[:, :, :, 4] < 0.5).unsqueeze(-1)   # (B,7,7,1)

    # 框回归 (只算有物体的格)
    box_loss = F.mse_loss(
        pred_boxes[:, :, :, :4] * obj_mask,
        targets[:, :, :, :4] * obj_mask,
        reduction='sum'
    ) / (obj_mask.sum() + 1e-6)

    # 置信度 (有物体→0, 无物体→1)
    conf_loss = F.mse_loss(pred_boxes[:, :, :, 4], targets[:, :, :, 4])

    # 分类 (只算有物体的格)
    cls_target = targets[:, :, :, 5].long()                 # (B,7,7)
    cls_loss = F.cross_entropy(
        pred_classes.reshape(-1, 3),
        cls_target.reshape(-1),
        reduction='sum'
    ) / (obj_mask.sum() + 1e-6)

    return box_loss + conf_loss + cls_loss


# ================================================================
#  4. 训练
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ToyDetectionDataset(n=500)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleYOLO().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)

    print(f"YOLO 训练 — 设备: {device} | 样本: {len(dataset)}")
    for epoch in range(30):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            pred_boxes, pred_classes = model(imgs)
            loss = yolo_loss(pred_boxes, pred_classes, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | loss: {total_loss/len(loader):.4f}")
    return model


# ================================================================
#  5. NMS（非极大值抑制，合并重叠框）
# ================================================================
def nms(boxes, scores, iou_thresh=0.5):
    """按得分排序，抑制和最高分框重叠过多的框"""
    if len(boxes) == 0: return []
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 0: break
        i = order[0].item(); keep.append(i)
        if order.numel() == 1: break
        # 算交并比
        x1 = torch.max(boxes[i,0]-boxes[i,2]/2, boxes[order[1:],0]-boxes[order[1:],2]/2)
        y1 = torch.max(boxes[i,1]-boxes[i,3]/2, boxes[order[1:],1]-boxes[order[1:],3]/2)
        x2 = torch.min(boxes[i,0]+boxes[i,2]/2, boxes[order[1:],0]+boxes[order[1:],2]/2)
        y2 = torch.min(boxes[i,1]+boxes[i,3]/2, boxes[order[1:],1]+boxes[order[1:],3]/2)
        inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
        union = boxes[i,2]*boxes[i,3] + boxes[order[1:],2]*boxes[order[1:],3] - inter
        iou = inter / (union + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


# ================================================================
#  6. 推理 + 可视化
# ================================================================
def predict_and_show(model, device):
    model.eval()
    dataset = ToyDetectionDataset(n=4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i in range(4):
        img_tensor, label = dataset[i]
        pred_boxes, pred_classes = model(img_tensor.unsqueeze(0).to(device))
        pred_boxes = pred_boxes[0].cpu()                     # (7,7,5)

        # 提取检测结果
        all_boxes, all_scores, all_cls = [], [], []
        for gy in range(GRID_S):
            for gx in range(GRID_S):
                conf = pred_boxes[gy, gx, 4].item()
                if conf > 0.5: continue                       # 0=有物体, >0.5=没有
                cx = (pred_boxes[gy,gx,0].item() + gx) / GRID_S
                cy = (pred_boxes[gy,gx,1].item() + gy) / GRID_S
                w  = pred_boxes[gy,gx,2].item() / GRID_S
                h  = pred_boxes[gy,gx,3].item() / GRID_S
                cls_id = pred_classes[0,gy,gx].argmax().item()
                all_boxes.append([cx, cy, w, h])
                all_scores.append(1.0)                         # 简化
                all_cls.append(cls_id)

        # NMS
        keep = nms(torch.tensor(all_boxes), torch.tensor(all_scores))
        final_boxes = [all_boxes[k] for k in keep]
        final_cls   = [all_cls[k] for k in keep]

        # 画图
        ax = axes[i]
        img = T.ToPILImage()(img_tensor)
        ax.imshow(img)
        for (cx, cy, w, h), cls_id in zip(final_boxes, final_cls):
            rect = patches.Rectangle(
                ((cx-w/2)*224, (cy-h/2)*224), w*224, h*224,
                linewidth=2, edgecolor=COLORS[cls_id], facecolor='none'
            )
            ax.add_patch(rect)
            ax.text((cx-w/2)*224, (cy-h/2)*224-5, CLASS_NAMES[cls_id],
                    color='black' if cls_id==1 else 'white', fontsize=10,
                    bbox=dict(facecolor=COLORS[cls_id], alpha=0.6))
        ax.axis('off')
        ax.set_title(f"图{i+1}: 检测到 {len(final_boxes)} 个")

    plt.tight_layout(); plt.savefig("yolo_result.png", dpi=100); plt.show()
    print("结果已保存: yolo_result.png")


# ================================================================
#  7. 运行
# ================================================================
if __name__ == "__main__":
    print("=== YOLO 目标检测 ===\n")
    model = train()
    predict_and_show(model, "cuda" if torch.cuda.is_available() else "cpu")
