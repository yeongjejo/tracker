from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor


# =========================
# Data classes
# =========================

@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]
    conf: float
    crop_bgr: np.ndarray


@dataclass
class Identity:
    gid: int
    feature: np.ndarray
    count: int
    last_frame: int


@dataclass
class DisplayItem:
    bbox: tuple[int, int, int, int]
    det_conf: float
    gid: int
    sim: float
    dist: float
    is_new: bool
    feature: np.ndarray
    base_feature: Optional[np.ndarray]
    crop_bgr: np.ndarray


# =========================
# Utility
# =========================

def parse_source(source: str) -> Union[int, str]:
    """
    "0"이면 웹캠 0번으로 처리.
    나머지는 영상 경로로 처리.
    """
    if source.isdigit():
        return int(source)
    return source


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: int = 0
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width - 1, x2 + margin)
    y2 = min(height - 1, y2 + margin)

    return x1, y1, x2, y2


def crop_person(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 5
) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h, margin)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    return crop


def put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1
) -> None:
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# =========================
# YOLO Detector
# =========================

class YoloPersonDetector:
    def __init__(self, model_path: str, conf: float):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        h, w = frame_bgr.shape[:2]

        results = self.model(
            frame_bgr,
            conf=self.conf,
            verbose=False,
            classes=[0]  # COCO person class
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[PersonDetection] = []

        for box, conf, cls_id in zip(boxes_xyxy, confs, classes):
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.astype(int).tolist()
            x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), w, h)

            crop = crop_person(frame_bgr, (x1, y1, x2, y2), margin=8)
            if crop is None:
                continue

            # 너무 작은 crop은 ReID 품질이 낮음
            ch, cw = crop.shape[:2]
            if ch < 40 or cw < 20:
                continue

            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                    crop_bgr=crop
                )
            )

        # 큰 사람부터 처리
        detections.sort(
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True
        )

        return detections


# =========================
# OSNet ReID
# =========================

class OSNetReID:
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: str = "",
        device: str = "cuda"
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device

        if model_path:
            model_path = str(Path(model_path))

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )

    @torch.no_grad()
    def extract(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.empty((0, 512), dtype=np.float32)

        # torchreid FeatureExtractor의 numpy 입력은 RGB 기준으로 넣는 것이 안전함
        crops_rgb = [
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            for crop in crops_bgr
        ]

        features = self.extractor(crops_rgb)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        features = np.asarray(features, dtype=np.float32)
        features = l2_normalize(features)

        return features


# =========================
# Gallery Matching
# =========================

class ReIDGallery:
    def __init__(
        self,
        match_thres: float = 0.2,
        smooth_alpha: float = 0.85,
        max_age: int = 300
    ):
        """
        match_thres:
            cosine distance 기준.
            dist = 1 - cosine_similarity.
            낮을수록 더 엄격함.

        smooth_alpha:
            기존 feature 평균을 얼마나 유지할지.
            0.85면 기존 85%, 신규 15%.

        max_age:
            너무 오래 안 나온 ID는 매칭 후보에서 제외.
        """
        self.match_thres = match_thres
        self.smooth_alpha = smooth_alpha
        self.max_age = max_age

        self.identities: dict[int, Identity] = {}
        self.next_gid = 1

    def match_or_create(
        self,
        feature: np.ndarray,
        frame_idx: int,
        used_gids: set[int]
    ) -> tuple[int, float, float, bool, Optional[np.ndarray]]:
        """
        return:
            gid, similarity, distance, is_new, base_feature_before_update
        """
        feature = l2_normalize(feature)

        best_gid = -1
        best_sim = -1.0
        best_dist = 999.0

        for gid, identity in self.identities.items():
            if gid in used_gids:
                continue

            if frame_idx - identity.last_frame > self.max_age:
                continue

            sim = cosine_similarity(feature, identity.feature)
            dist = 1.0 - sim

            if dist < best_dist:
                best_dist = dist
                best_sim = sim
                best_gid = gid

        print(self.match_thres)
        if best_gid != -1 and best_dist <= self.match_thres:
            identity = self.identities[best_gid]

            base_feature = identity.feature.copy()

            updated = (
                self.smooth_alpha * identity.feature
                + (1.0 - self.smooth_alpha) * feature
            )
            identity.feature = l2_normalize(updated)
            identity.count += 1
            identity.last_frame = frame_idx

            return best_gid, best_sim, best_dist, False, base_feature

        gid = self.next_gid
        self.next_gid += 1

        self.identities[gid] = Identity(
            gid=gid,
            feature=feature.copy(),
            count=1,
            last_frame=frame_idx
        )

        return gid, 1.0, 0.0, True, None

    def cleanup(self, frame_idx: int) -> None:
        remove_keys = []

        for gid, identity in self.identities.items():
            if frame_idx - identity.last_frame > self.max_age:
                remove_keys.append(gid)

        for gid in remove_keys:
            del self.identities[gid]


# =========================
# Feature Visualization
# =========================

def reduce_feature_bins(feature: np.ndarray, bins: int = 64) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).flatten()

    if feature.size == 0:
        return np.zeros((bins,), dtype=np.float32)

    if feature.size < bins:
        out = np.zeros((bins,), dtype=np.float32)
        out[:feature.size] = feature
        return out

    step = feature.size // bins
    values = []

    for i in range(bins):
        start = i * step
        end = (i + 1) * step if i < bins - 1 else feature.size
        values.append(float(np.mean(feature[start:end])))

    return np.asarray(values, dtype=np.float32)


def draw_feature_bar(
    image: np.ndarray,
    feature: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    bins: int = 64
) -> None:
    values = reduce_feature_bins(feature, bins=bins)

    max_abs = float(np.max(np.abs(values))) + 1e-6
    values = values / max_abs

    cv2.rectangle(image, (x, y), (x + w, y + h), (35, 35, 35), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (120, 120, 120), 1)

    put_text(image, title, (x, y - 8), (230, 230, 230), 0.48, 1)

    center_y = y + h // 2
    bar_w = max(1, w // bins)

    cv2.line(
        image,
        (x, center_y),
        (x + w, center_y),
        (150, 150, 150),
        1
    )

    for i, value in enumerate(values):
        bx = x + i * bar_w
        bh = int(abs(value) * (h // 2 - 5))

        if value >= 0:
            y1 = center_y - bh
            y2 = center_y
            color = (80, 220, 80)
        else:
            y1 = center_y
            y2 = center_y + bh
            color = (80, 80, 230)

        cv2.rectangle(
            image,
            (bx, y1),
            (min(bx + bar_w - 1, x + w), y2),
            color,
            -1
        )

def draw_feature_heatmap(
    image: np.ndarray,
    feature: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str
) -> None:
    img_h, img_w = image.shape[:2]

    # 그릴 위치가 화면 밖이면 return
    if x >= img_w or y >= img_h:
        return

    # 화면 안쪽으로 그릴 크기 보정
    draw_w = min(w, img_w - x)
    draw_h = min(h, img_h - y)

    if draw_w <= 0 or draw_h <= 0:
        return

    feature = np.asarray(feature, dtype=np.float32).flatten()

    if feature.size < 512:
        padded = np.zeros((512,), dtype=np.float32)
        padded[:feature.size] = feature
        feature = padded
    else:
        feature = feature[:512]

    f_min = float(np.min(feature))
    f_max = float(np.max(feature))

    norm = (feature - f_min) / (f_max - f_min + 1e-6)
    heat = (norm.reshape(16, 32) * 255).astype(np.uint8)

    heat = cv2.resize(
        heat,
        (draw_w, draw_h),
        interpolation=cv2.INTER_NEAREST
    )

    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)

    image[y:y + draw_h, x:x + draw_w] = heat_color

    cv2.rectangle(
        image,
        (x, y),
        (x + draw_w, y + draw_h),
        (180, 180, 180),
        1
    )

    put_text(
        image,
        title,
        (x, y - 8),
        (230, 230, 230),
        0.48,
        1
    )


def draw_diff_bar(
    image: np.ndarray,
    base_feature: np.ndarray,
    current_feature: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int
) -> None:
    diff = np.abs(
        reduce_feature_bins(base_feature, bins=64)
        - reduce_feature_bins(current_feature, bins=64)
    )

    max_val = float(np.max(diff)) + 1e-6
    diff = diff / max_val

    cv2.rectangle(image, (x, y), (x + w, y + h), (35, 35, 35), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (120, 120, 120), 1)

    put_text(image, "Feature Difference", (x, y - 8), (230, 230, 230), 0.48, 1)

    bins = len(diff)
    bar_w = max(1, w // bins)

    for i, value in enumerate(diff):
        bx = x + i * bar_w
        bh = int(value * (h - 8))

        cv2.rectangle(
            image,
            (bx, y + h - bh),
            (min(bx + bar_w - 1, x + w), y + h - 2),
            (0, 180, 255),
            -1
        )


def draw_crop_thumbnail(
    image: np.ndarray,
    crop_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int
) -> None:
    cv2.rectangle(image, (x, y), (x + w, y + h), (45, 45, 45), -1)

    if crop_bgr is None or crop_bgr.size == 0:
        put_text(image, "No Crop", (x + 10, y + 30), (200, 200, 200), 0.5, 1)
        return

    ch, cw = crop_bgr.shape[:2]

    scale = min(w / cw, h / ch)
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))

    resized = cv2.resize(crop_bgr, (nw, nh))

    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2

    image[oy:oy + nh, ox:ox + nw] = resized
    cv2.rectangle(image, (x, y), (x + w, y + h), (180, 180, 180), 1)


def make_canvas(
    frame_bgr: np.ndarray,
    panel_width: int = 460,
    min_canvas_height: int = 880
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    canvas_h = max(h, min_canvas_height)
    canvas_w = w + panel_width

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (22, 22, 22)

    # 원본 프레임 영역
    canvas[:h, :w] = frame_bgr

    # 원본 프레임 아래 남는 영역
    if canvas_h > h:
        canvas[h:canvas_h, :w] = (10, 10, 10)

    # 구분선
    cv2.line(canvas, (w, 0), (w, canvas_h), (90, 90, 90), 1)

    return canvas


def draw_panel(
    canvas: np.ndarray,
    frame_width: int,
    selected: Optional[DisplayItem],
    gallery_count: int,
    frame_idx: int,
    fps_text: str
) -> None:
    px = frame_width + 18
    py = 32

    put_text(canvas, "OSNet Feature Vector Demo", (px, py), (255, 255, 255), 0.65, 2)
    py += 28

    put_text(canvas, f"Frame: {frame_idx}", (px, py), (210, 210, 210), 0.5, 1)
    py += 22

    put_text(canvas, f"Gallery IDs: {gallery_count}", (px, py), (210, 210, 210), 0.5, 1)
    py += 22

    put_text(canvas, fps_text, (px, py), (210, 210, 210), 0.5, 1)
    py += 32

    if selected is None:
        put_text(canvas, "No person detected", (px, py), (120, 120, 255), 0.6, 2)
        return

    status = "NEW" if selected.is_new else "MATCH"
    status_color = (0, 200, 255) if selected.is_new else (80, 220, 80)

    put_text(canvas, f"Selected GID: {selected.gid} [{status}]", (px, py), status_color, 0.6, 2)
    py += 24

    put_text(canvas, f"YOLO Conf: {selected.det_conf:.3f}", (px, py), (220, 220, 220), 0.5, 1)
    py += 22

    put_text(canvas, f"Cos Sim: {selected.sim:.3f}", (px, py), (220, 220, 220), 0.5, 1)
    py += 22

    put_text(canvas, f"Cos Dist: {selected.dist:.3f}", (px, py), (220, 220, 220), 0.5, 1)
    py += 28

    draw_crop_thumbnail(canvas, selected.crop_bgr, px, py, 120, 160)

    put_text(canvas, "Person Crop", (px + 140, py + 30), (230, 230, 230), 0.55, 1)
    put_text(canvas, "Feature shape: 512", (px + 140, py + 58), (230, 230, 230), 0.5, 1)
    put_text(canvas, "Vector values are", (px + 140, py + 86), (190, 190, 190), 0.48, 1)
    put_text(canvas, "not RGB values.", (px + 140, py + 110), (190, 190, 190), 0.48, 1)

    py += 200

    draw_feature_bar(
        canvas,
        selected.feature,
        px,
        py,
        410,
        70,
        "Current Feature Bar"
    )
    py += 110

    draw_feature_heatmap(
        canvas,
        selected.feature,
        px,
        py,
        410,
        90,
        "Current Feature Heatmap"
    )
    py += 130

    if selected.base_feature is not None:
        draw_feature_bar(
            canvas,
            selected.base_feature,
            px,
            py,
            410,
            70,
            "Base/Gallery Feature Bar"
        )
        py += 110

        draw_diff_bar(
            canvas,
            selected.base_feature,
            selected.feature,
            px,
            py,
            410,
            55
        )
    else:
        put_text(canvas, "Base feature: none", (px, py), (120, 120, 255), 0.55, 1)
        put_text(canvas, "New identity added to gallery", (px, py + 25), (120, 120, 255), 0.5, 1)


def draw_detection_result(
    frame_bgr: np.ndarray,
    item: DisplayItem
) -> None:
    x1, y1, x2, y2 = item.bbox

    if item.is_new:
        color = (0, 200, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    label = f"GID:{item.gid} Dist:{item.dist:.3f} Sim:{item.sim:.3f}"

    label_y = max(20, y1 - 8)

    cv2.rectangle(
        frame_bgr,
        (x1, label_y - 18),
        (min(x1 + 330, frame_bgr.shape[1] - 1), label_y + 4),
        color,
        -1
    )

    put_text(
        frame_bgr,
        label,
        (x1 + 4, label_y),
        (0, 0, 0),
        0.5,
        1
    )


# =========================
# Main
# =========================

def run(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 30.0

    detector = YoloPersonDetector(
        model_path=args.yolo,
        conf=args.det_conf
    )

    reid = OSNetReID(
        model_name=args.reid_model,
        model_path=args.reid_weights,
        device=args.device
    )

    gallery = ReIDGallery(
        match_thres=args.match_thres,
        smooth_alpha=args.smooth_alpha,
        max_age=args.max_age
    )

    writer = None
    frame_idx = 0

    prev_time = cv2.getTickCount()
    smoothed_fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        detections = detector.detect(frame)

        display_items: list[DisplayItem] = []
        used_gids: set[int] = set()

        if detections:
            crops = [det.crop_bgr for det in detections]
            features = reid.extract(crops)

            for det, feature in zip(detections, features):
                gid, sim, dist, is_new, base_feature = gallery.match_or_create(
                    feature=feature,
                    frame_idx=frame_idx,
                    used_gids=used_gids
                )

                used_gids.add(gid)

                item = DisplayItem(
                    bbox=det.bbox,
                    det_conf=det.conf,
                    gid=gid,
                    sim=sim,
                    dist=dist,
                    is_new=is_new,
                    feature=feature,
                    base_feature=base_feature,
                    crop_bgr=det.crop_bgr
                )

                display_items.append(item)

        # gallery.cleanup(frame_idx)

        for item in display_items:
            draw_detection_result(frame, item)

        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now

        if dt > 0:
            current_fps = 1.0 / dt
            if smoothed_fps == 0:
                smoothed_fps = current_fps
            else:
                smoothed_fps = 0.9 * smoothed_fps + 0.1 * current_fps

        fps_text = f"FPS: {smoothed_fps:.1f}"

        selected = display_items[0] if display_items else None

        h, w = frame.shape[:2]
        canvas = make_canvas(frame, panel_width=args.panel_width)

        draw_panel(
            canvas=canvas,
            frame_width=w,
            selected=selected,
            gallery_count=len(gallery.identities),
            frame_idx=frame_idx,
            fps_text=fps_text
        )

        if writer is None and args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                fps,
                (canvas.shape[1], canvas.shape[0])
            )

        if writer is not None:
            writer.write(canvas)

        cv2.imshow("OSNet Feature Vector Visualization", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

        if key == ord("r"):
            gallery = ReIDGallery(
                match_thres=args.match_thres,
                smooth_alpha=args.smooth_alpha,
                max_age=args.max_age
            )
            print("[INFO] Gallery reset")

    cap.release()

    if writer is not None:
        writer.release()
        print(f"[INFO] Saved output video: {args.output}")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO + OSNet feature vector visualization demo"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index or video path. Example: 0 or input.mp4"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="osnet_feature_demo_output.mp4",
        help="Output mp4 path. Empty string disables saving."
    )

    parser.add_argument(
        "--yolo",
        type=str,
        default="yolo11n.pt",
        help="YOLO model path. Example: yolo11n.pt, yolov8n.pt"
    )

    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.45,
        help="YOLO person detection confidence threshold"
    )

    parser.add_argument(
        "--reid-model",
        type=str,
        default="osnet_x1_0",
        help="Torchreid model name. Example: osnet_x1_0"
    )

    parser.add_argument(
        "--reid-weights",
        type=str,
        default="",
        help="Optional OSNet weight path. Example: osnet_x1_0_market.pth.tar"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device"
    )

    parser.add_argument(
        "--match-thres",
        type=float,
        default=0.2,
        help="Cosine distance threshold. Lower is stricter. Typical: 0.25~0.45"
    )

    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.85,
        help="Gallery feature smoothing weight"
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=300,
        help="Frames to keep inactive identity"
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=460,
        help="Right visualization panel width"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)