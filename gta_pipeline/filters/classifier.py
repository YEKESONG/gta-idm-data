"""可插拔 ML 帧分类器接口。

启发式规则覆盖了"明显"的坏画面；遇到难分的情况（光照怪异、改装 HUD、
第一/第三人称切换等），可以训练一个轻量 CNN 来兜底。本模块提供：

1) FrameClassifier 协议：任何实现了 predict() 的对象都能接进来；
2) TorchResNetClassifier：基于 torchvision ResNet18 的参考实现，从 ckpt 加载；
3) ClassifierDetector：把分类器适配成标准 FrameDetector，与启发式无缝混用。

未提供权重（classifier_ckpt 为空）时，ClassifierDetector 会自我禁用
（disabled=True），build_detectors 会自动跳过它，因此默认管线纯启发式即可跑。

约定：分类器输出"该帧不是有效 gameplay 的概率" prob_not_gameplay ∈ [0,1]。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .base import DetectorResult, FrameDetector, register

# 训练分类器时建议的类别（仅作参考，可自行调整）。
DEFAULT_CLASSES = ["gameplay", "death", "menu", "loading"]


@runtime_checkable
class FrameClassifier(Protocol):
    """帧分类器协议。实现它即可接入框架。"""

    def predict(self, frame: np.ndarray) -> float:
        """输入 BGR 帧，返回'不是有效 gameplay'的概率 ∈ [0,1]。"""
        ...


class TorchResNetClassifier:
    """基于 torchvision ResNet18 的参考分类器。

    期望 ckpt 是用 torch.save(model.state_dict()) 存的权重，
    分类头维度 = len(classes)，第 0 类约定为 "gameplay"。
    """

    def __init__(self, ckpt_path: str, classes: list[str] | None = None, device: str | None = None):
        import torch
        from torchvision import models, transforms

        self.classes = classes or DEFAULT_CLASSES
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(self.classes))
        state = torch.load(ckpt_path, map_location=self.device)
        net.load_state_dict(state)
        net.eval().to(self.device)
        self._net = net
        self._torch = torch
        self._tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, frame: np.ndarray) -> float:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._torch.no_grad():
            x = self._tf(rgb).unsqueeze(0).to(self.device)
            probs = self._torch.softmax(self._net(x), dim=1)[0]
        gameplay_idx = self.classes.index("gameplay") if "gameplay" in self.classes else 0
        return float(1.0 - probs[gameplay_idx].item())


@register
class ClassifierDetector(FrameDetector):
    """把 FrameClassifier 包装成标准检测器。"""

    name = "classifier"

    def __init__(
        self,
        classifier_ckpt: str | None = None,
        classifier_threshold: float = 0.5,
        classifier: FrameClassifier | None = None,
    ) -> None:
        self.threshold = classifier_threshold
        self.disabled = False
        if classifier is not None:
            self._clf: FrameClassifier | None = classifier
        elif classifier_ckpt:
            self._clf = TorchResNetClassifier(classifier_ckpt)
        else:
            # 没给权重也没给现成分类器 -> 禁用，build_detectors 会跳过。
            self._clf = None
            self.disabled = True

    def __call__(self, frame: np.ndarray) -> DetectorResult:
        assert self._clf is not None  # disabled 时不会被调用
        prob = self._clf.predict(frame)
        label = "non_gameplay" if prob >= self.threshold else "gameplay"
        return DetectorResult(self.name, label, prob, {"prob_not_gameplay": round(prob, 3)})
