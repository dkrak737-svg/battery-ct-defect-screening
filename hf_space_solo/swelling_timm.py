# -*- coding: utf-8 -*-
"""
swelling_timm.py
본인이 학습한 swelling 분류기(timm efficientnet_b0) 추론.

학습 레시피(vision/train_swelling.py 와 동일):
  - timm efficientnet_b0, num_classes=2 (nonswell=0, swell=1)
  - 입력 512 레터박스(crop 금지) → RGB → ImageNet 정규화
  - P(swell)=softmax[:,1], 슬라이스 thr 0.5, 배터리 k=1(1장↑ swell → 부풂)
"""
import numpy as np
import torch
from torchvision import transforms

import recipe

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SwellingClassifier:
    def __init__(self, weight_path, imgsz=512, backbone="efficientnet_b0"):
        import timm
        self.imgsz = imgsz
        self.model = timm.create_model(backbone, pretrained=False, num_classes=2)
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval().to(DEVICE)
        self.tf = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),     # 512 레터박스라 사실상 no-op(안전장치)
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @torch.no_grad()
    def prob_swell(self, gray_imgs, chunk=64):
        """회색('L') 크롭 이미지 리스트 → 각 슬라이스 P(swell) ndarray."""
        out = []
        for s in range(0, len(gray_imgs), chunk):
            xs = []
            for g in gray_imgs[s:s + chunk]:
                lb = recipe.letterbox(g, self.imgsz).convert("RGB")
                xs.append(self.tf(lb))
            x = torch.stack(xs).to(DEVICE)
            logit = self.model(x)
            p = torch.softmax(logit.float(), 1)[:, 1]
            out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.array([])
