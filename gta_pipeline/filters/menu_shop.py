"""菜单 / 商店 / 暂停界面检测。

这类画面要丢弃，因为它们没有有效的"移动 + 开火"动作信号。
GTA V 的暂停菜单、地图、武器商店共同特征：
- 大面积**平坦区域**（纯色面板、半透明遮罩），自然纹理少；
- 常有一条贯穿屏幕的纯色侧栏/顶栏。

启发式：统计"平坦像素"占比（局部梯度很小的像素）。gameplay 画面
有丰富纹理（街道、建筑、人物），平坦占比低；菜单则很高。
预留模板钩子：把商店图标/菜单标志放 assets/templates/menu/ 下可增强判定。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import DetectorResult, FrameDetector, register


@register
class MenuShopDetector(FrameDetector):
    name = "menu_shop"

    def __init__(
        self,
        flat_ratio_threshold: float = 0.6,  # 平坦像素占比超过此值判为菜单
        gradient_threshold: float = 12.0,  # 梯度幅度低于此值算"平坦"
        template_dir: str = "assets/templates/menu",
        template_match_threshold: float = 0.75,
    ) -> None:
        self.flat_ratio_threshold = flat_ratio_threshold
        self.gradient_threshold = gradient_threshold
        self.template_match_threshold = template_match_threshold
        self._templates = _load_gray_templates(template_dir)

    def __call__(self, frame: np.ndarray) -> DetectorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sobel 梯度幅度近似局部纹理强度。
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        flat_ratio = float((mag < self.gradient_threshold).mean())

        score = flat_ratio if flat_ratio >= self.flat_ratio_threshold else 0.0
        info = {"flat_ratio": round(flat_ratio, 3)}

        if self._templates:
            tm = _best_template_score(gray, self._templates)
            info["template"] = round(tm, 3)
            if tm >= self.template_match_threshold:
                score = max(score, tm)

        label = "menu" if score >= self.flat_ratio_threshold else "gameplay"
        return DetectorResult(self.name, label, score, info)


def _load_gray_templates(template_dir: str) -> list[np.ndarray]:
    d = Path(template_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.append(img)
    return out


def _best_template_score(gray: np.ndarray, templates: list[np.ndarray]) -> float:
    best = 0.0
    for tpl in templates:
        if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best
