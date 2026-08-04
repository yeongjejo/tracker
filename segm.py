import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import onnxruntime as ort

from datetime import datetime

# ==============================
# 설정값
# ==============================
IMAGE_PATH = "color/yellow.png"          # 입력 이미지 경로
MODEL_PATH = "yolo26n-seg.pt"  # 또는 "yolov8n-seg.pt"
OUTPUT_DIR = "color/yellow"

CONF_THRES = 0.35                 # 사람 검출 신뢰도
MASK_THRES = 0.5                  # segmentation mask threshold

SAVE_EACH_PERSON_CROP = True      # 사람별 crop 저장
SAVE_FULL_IMAGE = False            # 전체 이미지 기준 배경 흰색 저장
SAVE_LARGEST_PERSON_ONLY = False  # 가장 큰 사람 1명만 처리할지 여부



model_path = str(Path("osnet_x1_0.onnx").resolve())
engine_cache_dir = str(Path("./osnet_trt_cache").resolve())

os.makedirs(engine_cache_dir, exist_ok=True)

available_providers = ort.get_available_providers()
print("Available providers:", available_providers)

if "TensorrtExecutionProvider" not in available_providers:
    raise RuntimeError(
        "TensorrtExecutionProvider를 사용할 수 없습니다. "
        "onnxruntime-gpu, CUDA 및 TensorRT 설치 상태를 확인하세요."
    )

providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_cache_dir,
            "trt_fp16_enable": True,
        }
    ),
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0
        }
    ),
    "CPUExecutionProvider"
]

session = ort.InferenceSession(
    model_path,
    providers=providers
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def save_feature(gid, features):

    if features is None or features.shape[0] == 0:
        print('저장할 feature가 없음')
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-5]

    file_path = f"./testfeature//reid_feature_id_{gid}_{timestamp}.npz"

    if not file_path:
        return

    save_path = Path(file_path)
    if save_path.suffix.lower() != ".npz":
        save_path = save_path.with_suffix(".npz")

    try:
        np.savez_compressed(
            str(save_path),
            format_version=np.array([1], dtype=np.int32),
            gid=np.array([gid], dtype=np.int32),
            features=np.ascontiguousarray(
                features,
                dtype=np.float32
            ),
            feature_count=np.array(
                [features.shape[0]],
                dtype=np.int32
            ),
            feature_dim=np.array(
                [features.shape[1]],
                dtype=np.int32
            ),
            model_name=np.array(["osnet_x1_0"], dtype="<U32"),
            saved_at=np.array(
                [datetime.now().isoformat(timespec="seconds")],
                dtype="<U32"
            )
        )
    except Exception as error:
        print('피쳐 저장 실피 : ', error)
        return




def normalize_feature_array(features):
    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValueError(
            f"Feature 배열은 [N, D] 형태여야 합니다: {features.shape}"
        )

    if features.shape[0] == 0:
        raise ValueError("Feature 데이터가 비어 있습니다.")

    if features.shape[1] != 512:
        raise ValueError(
            "Feature 차원이 올바르지 않습니다. "
            f"expected=512, actual={features.shape[1]}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("Feature 데이터에 NaN 또는 Inf가 포함되어 있습니다.")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12

    if not np.all(valid):
        features = features[valid]
        norms = norms[valid]

    if features.shape[0] == 0:
        raise ValueError("유효한 Feature 벡터가 없습니다.")

    return np.ascontiguousarray(
        features / np.maximum(norms, 1e-12),
        dtype=np.float32
    )

def preprocess(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None

    # OpenCV BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # OSNet 입력: width=128, height=256
    img_rgb = cv2.resize(
        img_rgb,
        (128, 256),
        interpolation=cv2.INTER_LINEAR
    )

    image = img_rgb.astype(np.float32) / 255.0

    # Torchreid 기본 ImageNet 정규화
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std

    # HWC → CHW
    image = image.transpose(2, 0, 1)

    # CHW → NCHW
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(
        image,
        dtype=np.float32
    )



def extract(img_bgr):
    features = extract_batch([img_bgr])

    if len(features) == 0:
        return None

    return features[0]


def extract_batch(images_bgr):
    tensors = []

    for image in images_bgr:
        tensor = preprocess(image)

        if tensor is not None:
            # preprocess 결과: [1, 3, 256, 128]
            # 배치 결합을 위해 첫 번째 축 제거
            tensors.append(tensor[0])

    if not tensors:
        return np.empty((0, 512), dtype=np.float32)

    # [N, 3, 256, 128]
    batch = np.stack(
        tensors,
        axis=0
    ).astype(np.float32)

    batch = np.ascontiguousarray(batch)

    features = session.run(
        [output_name],
        {
            input_name: batch
        }
    )[0]

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # 혹시 출력이 [N, 512, 1, 1] 형태인 경우 대비
    features = features.reshape(features.shape[0], -1)

    # 각 Feature L2 정규화
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    features = features / np.maximum(norms, 1e-12)

    return features


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

def main(id, path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMAGE_PATH = path

    img = cv2.imread(str(IMAGE_PATH))

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
        buffer = []
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

            # save_path = os.path.join(OUTPUT_DIR, f"person_{idx}_crop_white.png")
            # cv2.imwrite(save_path, output_crop)
            # print(f"crop 저장 완료: {save_path}")

            feat = extract(output_crop) #피쳐 저장
            buffer.append(feat)

        normalized = normalize_feature_array(buffer)
        np_normalized = np.asarray(
            normalized,
            dtype=np.float32
        ).copy()

        save_feature(id, np_normalized)

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
    color_list = ['orange', 'bluesky', 'green', 'yellow', 'blue', 'purple', 'pink', 'black']
    #
    # for i, color in enumerate(color_list):
    #     main(i, color)

    for i, color in enumerate(color_list):
        folder_path = Path("./color/"+color)
        for path in folder_path.glob("*.png"):
            main(i, path)