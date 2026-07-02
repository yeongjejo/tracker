import os
import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# 설정값
# ==============================

IMAGE_PATH = "img.png"          # 입력 이미지 경로
MODEL_PATH = "yolo11n-seg2.pt"  # 또는 "yolov8n-seg.pt"
OUTPUT_DIR = "output"

CONF_THRES = 0.35                 # 사람 검출 신뢰도
MASK_THRES = 0.5                  # segmentation mask threshold

SAVE_EACH_PERSON_CROP = True      # 사람별 crop 저장
SAVE_FULL_IMAGE = True            # 전체 이미지 기준 배경 흰색 저장
SAVE_LARGEST_PERSON_ONLY = False  # 가장 큰 사람 1명만 처리할지 여부


# ==============================
# 유틸 함수
# ==============================

def make_odd_kernel(value):
    value = int(value)
    if value < 3:
        value = 3
    if value % 2 == 0:
        value += 1
    return value


def smooth_mask(binary_mask, blur_size=21):
    """
    마스크 경계를 부드럽게 만들기 위한 soft alpha mask 생성
    """
    mask = (binary_mask * 255).astype(np.uint8)

    # 작은 구멍 메우기
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 경계 부드럽게
    # blur_size = make_odd_kernel(blur_size)
    # mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # 0~1 alpha로 변환
    alpha = mask.astype(np.float32) / 255.0
    return alpha


def apply_background_white(image, person_mask):
    """
    사람 영역은 원본 유지, 배경은 흰색으로 처리
    """
    alpha = smooth_mask(person_mask, blur_size=21)
    alpha_3ch = np.repeat(alpha[:, :, None], 3, axis=2)

    white_bg = np.ones_like(image, dtype=np.uint8) * 255

    output = image.astype(np.float32) * alpha_3ch + white_bg.astype(np.float32) * (1 - alpha_3ch)
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def clip_box(box, img_w, img_h):
    """
    bbox가 이미지 범위를 벗어나지 않도록 보정
    """
    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))

    return x1, y1, x2, y2


# ==============================
# 메인 처리
# ==============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {IMAGE_PATH}")

    img_h, img_w = img.shape[:2]

    # YOLO segmentation 모델 로드
    model = YOLO(MODEL_PATH)

    # COCO 기준 person class = 0
    results = model.predict(
        source=img,
        classes=[0],
        conf=CONF_THRES,
        retina_masks=True
    )

    result = results[0]

    if result.masks is None or result.boxes is None:
        print("사람이 검출되지 않았습니다.")
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    print(f"검출된 사람 수: {len(boxes)}")

    # 가장 큰 사람만 처리
    if SAVE_LARGEST_PERSON_ONLY and len(boxes) > 0:
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

        largest_idx = int(np.argmax(areas))
        boxes = [boxes[largest_idx]]
        masks = [masks[largest_idx]]

        print(f"가장 큰 사람 index: {largest_idx}")

    # ==============================
    # 1. 사람별 bbox crop 후 배경 흰색 처리
    # ==============================
    if SAVE_EACH_PERSON_CROP:
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2 = clip_box(box, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            # bbox crop
            crop_img = img[y1:y2, x1:x2].copy()

            # mask crop
            crop_mask = mask[y1:y2, x1:x2]

            # mask 이진화
            crop_mask = (crop_mask > MASK_THRES).astype(np.uint8)

            # crop 영역 안에서 배경 흰색 처리
            output_crop = apply_background_white(
                image=crop_img,
                person_mask=crop_mask
            )

            save_path = os.path.join(OUTPUT_DIR, f"person_{idx}_crop_white.png")
            cv2.imwrite(save_path, output_crop)

            print(f"crop 저장 완료: {save_path}")

    # ==============================
    # 2. 전체 이미지 기준 사람은 원본, 배경은 흰색 처리
    # ==============================
    if SAVE_FULL_IMAGE:
        full_person_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for mask in masks:
            binary_mask = (mask > MASK_THRES).astype(np.uint8)
            full_person_mask = np.maximum(full_person_mask, binary_mask)

        output_full = apply_background_white(
            image=img,
            person_mask=full_person_mask
        )

        save_path = os.path.join(OUTPUT_DIR, "full_image_person_clear_background_white.png")
        cv2.imwrite(save_path, output_full)

        print(f"전체 이미지 저장 완료: {save_path}")


if __name__ == "__main__":
    main()