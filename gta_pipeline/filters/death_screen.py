"""死亡 / 任务结算画面检测（GTA V 的 WASTED / BUSTED / MISSION FAILED）。

GTA V 死亡画面的两个稳健视觉特征：
1) 整屏强烈去饱和（接近灰度，偶带暖色调）——这是引擎的特效；
2) 屏幕中央有一行很大的高对比文字（WASTED 等）。

启发式：低饱和 + 中央条带高对比文字 → 判为 death。
若装了 pytesseract，则进一步 OCR 中央条带做关键词确认，提高精度。
还预留模板匹配钩子：把死亡画面截图放到 assets/templates/death/ 下即可启用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import DetectorResult, FrameDetector, register

_KEYWORDS = ("WASTED", "BUSTED", "MISSION FAILED", "MISSION PASSED", "DIED")


@register
class DeathScreenDetector(FrameDetector):
    name = "death_screen"

    def __init__(
        self,
        sat_threshold: float = 35.0,  # 平均饱和度低于此值视为"去饱和"
        text_contrast_threshold: float = 55.0,  # 中央条带对比度（标准差）阈值
        template_dir: str = "assets/templates/death",
        template_match_threshold: float = 0.7,
    ) -> None:
        self.sat_threshold = sat_threshold
        self.text_contrast_threshold = text_contrast_threshold
        self.template_match_threshold = template_match_threshold
        self._templates = _load_templates(template_dir)
        self._ocr = _try_load_ocr()

    def __call__(self, frame: np.ndarray) -> DetectorResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        desaturated = mean_sat < self.sat_threshold

        # 取中央水平条带（大字一般在画面中部）。
        h, w = frame.shape[:2]
        band = frame[int(h * 0.35) : int(h * 0.65), :]
        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        contrast = float(gray_band.std())
        has_big_text = contrast > self.text_contrast_threshold

        score = 0.0
        info = {"mean_sat": mean_sat, "band_contrast": contrast}

        if desaturated and has_big_text:
            score = 0.6
            # OCR 关键词确认（可选，命中则强烈置信）。
            if self._ocr is not None:
                text = self._ocr(gray_band).upper()
                info["ocr"] = text[:50]
                if any(k in text for k in _KEYWORDS):
                    score = 0.95

        # 模板匹配（可选，与启发式取较大者）。
        if self._templates:
            tm = _best_template_score(frame, self._templates)
            info["template"] = round(tm, 3)
            if tm >= self.template_match_threshold:
                score = max(score, tm)

        label = "death" if score >= 0.6 else "gameplay"
        return DetectorResult(self.name, label, score, info)


def _load_templates(template_dir: str) -> list[np.ndarray]:
    d = Path(template_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.append(img)
    return out


def _best_template_score(frame: np.ndarray, templates: list[np.ndarray]) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = 0.0
    for tpl in templates:
        if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best


def _try_load_ocr():
    try:
        import pytesseract  # noqa: F401

        def _ocr(gray_img: np.ndarray) -> str:
            import pytesseract

            return pytesseract.image_to_string(gray_img)

        return _ocr
    except Exception:
        return None  # 没装就静默退回纯启发式
