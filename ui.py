import random
import sys
import cv2
import numpy as np
import torch
import time
import threading

from ultralytics import YOLO
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import os
from pathlib import Path
import onnxruntime as ort

from torchreid.reid.utils.feature_extractor import FeatureExtractor

from datetime import datetime


# =========================
# REID MODEL
# =========================
class ReIDModel:
    def __init__(self, model_path="osnet_x1_0.onnx", engine_cache_dir="./osnet_trt_cache"):
        model_path = str(Path(model_path).resolve())
        engine_cache_dir = str(Path(engine_cache_dir).resolve())

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

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("OSNet providers:", self.session.get_providers())
        print("OSNet input:", self.input_name)
        print("OSNet output:", self.output_name)

    def preprocess(self, img_bgr):
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

    def extract(self, img_bgr):
        features = self.extract_batch([img_bgr])

        if len(features) == 0:
            return None

        return features[0]

    def extract_batch(self, images_bgr):
        tensors = []

        for image in images_bgr:
            tensor = self.preprocess(image)

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

        features = self.session.run(
            [self.output_name],
            {
                self.input_name: batch
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

# =========================
# MEMORY
# =========================
class GlobalMemory:
    FEATURE_DIM = 512

    def __init__(self):
        self.base_data = {}  # id -> [N, 512] base features
        self.real_time_data = {i: [] for i in range(8)}
        self.unknown_data = {}
        self.real_time_count = [100 for _ in range(8)]

        # UI의 저장/불러오기와 Worker의 매칭이 동시에 접근할 수 있으므로 보호합니다.
        self.data_lock = threading.RLock()

    def subtract_real_time_cont(self, id_list):
        with self.data_lock:
            for gid in range(8):
                if gid not in id_list:
                    self.real_time_count[gid] = max(
                        0,
                        self.real_time_count[gid] - 1
                    )

                    if self.real_time_count[gid] == 0:
                        self.real_time_data[gid] = []
                        print("실시간 feature 정리 : ", gid)
                else:
                    self.real_time_count[gid] = 100

    def subtract_unknown_data_cont(self):
        with self.data_lock:
            for key, value in list(self.unknown_data.items()):
                if value[1] <= 0:
                    print("unknown_data 정리")
                    del self.unknown_data[key]
                else:
                    self.unknown_data[key][1] -= 1

    @staticmethod
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

        if features.shape[1] != GlobalMemory.FEATURE_DIM:
            raise ValueError(
                "Feature 차원이 올바르지 않습니다. "
                f"expected={GlobalMemory.FEATURE_DIM}, actual={features.shape[1]}"
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

    def add(self, gid, features):
        normalized = self.normalize_feature_array(features)

        with self.data_lock:
            self.base_data[int(gid)] = normalized.copy()
            self.real_time_data[int(gid)] = []
            self.real_time_count[int(gid)] = 100

    def clear_id(self, gid):
        gid = int(gid)

        with self.data_lock:
            self.base_data.pop(gid, None)
            self.real_time_data[gid] = []
            self.real_time_count[gid] = 100

    def get_base_features_copy(self, gid):
        gid = int(gid)

        with self.data_lock:
            features = self.base_data.get(gid)

            if features is None:
                return None

            return np.asarray(
                features,
                dtype=np.float32
            ).copy()

    def set_base_features(self, gid, features):
        gid = int(gid)
        normalized = self.normalize_feature_array(features)

        with self.data_lock:
            self.base_data[gid] = normalized.copy()

            # 이전 실행에서 생성된 보정 Feature가 새 파일에 섞이지 않도록 초기화
            self.real_time_data[gid] = []
            self.real_time_count[gid] = 100
            self.unknown_data.clear()

        return normalized.shape[0]

    def get_feature_count(self, gid):
        features = self.get_base_features_copy(gid)
        return 0 if features is None else int(features.shape[0])

    def match(self, feature):
        # UI에서 Feature 파일을 불러오거나 삭제하는 중에 dict가 변경되지 않도록 보호
        with self.data_lock:
            return self._match_locked(feature)

    def _match_locked(self, feature):
        best_id = None
        best_score = -1

        un_best_id = None
        un_best_score = 0

        for gid, value in self.unknown_data.items():
            un_feature = value[0].copy()
            un_feature = un_feature / max(
                np.linalg.norm(un_feature),
                1e-12
            )
            score = np.dot(feature, un_feature)

            if score > un_best_score:
                un_best_score = score
                un_best_id = gid

        if un_best_score > 0.75 and un_best_id is not None:
            smooth_alpha = 0.85
            updated = (
                smooth_alpha * self.unknown_data[un_best_id][0]
                + (1.0 - smooth_alpha) * feature
            )
            new_realtime_feature = updated / max(
                np.linalg.norm(updated),
                1e-12
            )
            self.unknown_data[un_best_id][0] = new_realtime_feature

        for gid, base_features in self.base_data.items():
            base_mean = np.mean(base_features, axis=0)
            base_mean = base_mean / max(
                np.linalg.norm(base_mean),
                1e-12
            )

            if len(self.real_time_data[gid]) > 0:
                real_time_rate = 0.8 * (
                    self.real_time_count[gid] / 100.0
                )
                base_rate = 1.0 - real_time_rate
                mean = (
                    base_mean * base_rate
                    + self.real_time_data[gid][0] * real_time_rate
                )
            else:
                mean = base_mean

            mean = mean / max(np.linalg.norm(mean), 1e-12)
            score = np.dot(feature, mean)

            if score > best_score:
                best_score = score
                best_id = gid

        if (
            best_id is not None
            and best_score > 0.75
            and best_score > un_best_score
        ):
            if len(self.real_time_data[best_id]) == 0:
                self.real_time_data[best_id].append(
                    feature / max(np.linalg.norm(feature), 1e-12)
                )
            else:
                smooth_alpha = 0.85
                updated = (
                    smooth_alpha * self.real_time_data[best_id][0]
                    + (1.0 - smooth_alpha) * feature
                )
                new_realtime_feature = updated / max(
                    np.linalg.norm(updated),
                    1e-12
                )
                self.real_time_data[best_id][0] = new_realtime_feature

            return best_id, best_score

        if len(self.unknown_data) == 0:
            num = random.randint(0, 9999)
            self.unknown_data[num] = [feature.copy(), 30]
        elif un_best_score <= 0.75:
            while True:
                num = random.randint(0, 9999)
                if num not in self.unknown_data:
                    self.unknown_data[num] = [feature.copy(), 30]
                    break

        return None, None


# =========================
# WORKER THREAD
# =========================
class Worker(QThread):
    update_frame = Signal(QImage)
    status_signal = Signal(str)
    source_changed_signal = Signal(str, str)
    playback_finished_signal = Signal()

    def __init__(self, memory):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        # =========================
        # INPUT SOURCE
        # =========================
        self.source_type = "camera"      # camera / video
        self.source_path = None
        self.source_name = "Camera 0"
        self.source_fps = 30.0
        self.source_frame_interval = 1.0 / self.source_fps
        self.last_source_frame = None
        self.pending_source = None
        self.source_lock = threading.Lock()

        # self.model = YOLO("yolo11n.pt")
        # self.model = YOLO("yolo11n-seg.pt")
        # self.model = YOLO("yolo26n-seg.pt")
        self.model = YOLO("yolo26n-seg.engine")

        self.reid = ReIDModel()

        self.memory = memory

        self.mode = "idle"      # idle / capture / live
        self.live_on = False

        self.target_id = None
        self.buffer = []

        self.mask_thres = 0.5

        self.output_full = None
        self.prev_time = time.perf_counter()
        self.fps = 0.0

        self.time_yolo = 0.0
        self.time_osnet = 0.0
        self.time_match = 0.0
        self.time_total = 0.0
        self.processing_fps = 0.0

        self.id_color = [
            (230, 159, 0),
            (86, 180, 233),
            (0, 158, 115),
            (240, 228, 66),
            (0, 114, 178),
            (213, 94, 0),
            (204, 121, 167),
            (0, 0, 0),
        ]

        # =========================
        # RAW VIDEO RECORDING
        # =========================
        # True일 때 Live 시작과 함께 원본 카메라 프레임을 녹화합니다.
        self.recording_enabled = True
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None
        self.recording_fps = 30.0
        self.recording_last_time = None
        self.recording_stop_requested = False
        self.running = True

        # 현재 실행 중인 파이썬 파일과 같은 폴더에 저장
        try:
            self.recording_dir = Path(__file__).resolve().parent
        except NameError:
            self.recording_dir = Path.cwd()

    # =========================
    # INPUT SOURCE CONTROL
    # =========================
    def request_video_source(self, video_path):
        """UI 스레드에서 영상 파일 변경을 요청합니다."""
        with self.source_lock:
            self.pending_source = ("video", str(video_path))

    def request_camera_source(self, camera_index=0):
        """UI 스레드에서 카메라 소스 변경을 요청합니다."""
        with self.source_lock:
            self.pending_source = ("camera", int(camera_index))

    def take_pending_source(self):
        with self.source_lock:
            pending = self.pending_source
            self.pending_source = None
        return pending

    def reset_runtime_stats(self):
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.time_yolo = 0.0
        self.time_osnet = 0.0
        self.time_match = 0.0
        self.time_total = 0.0
        self.processing_fps = 0.0

    def apply_pending_source(self):
        """VideoCapture 교체는 Worker 스레드 내부에서만 수행합니다."""
        pending = self.take_pending_source()
        if pending is None:
            return

        source_type, source_value = pending

        if source_type == "video":
            new_cap = cv2.VideoCapture(source_value)

            if not new_cap.isOpened():
                new_cap.release()
                self.status_signal.emit(
                    f"VIDEO OPEN FAILED: {source_value}"
                )
                return

            # 첫 프레임을 미리 읽어 정지 화면으로 표시한 뒤 위치를 처음으로 복구
            ret, preview_frame = new_cap.read()
            if not ret or preview_frame is None:
                new_cap.release()
                self.status_signal.emit(
                    f"VIDEO FRAME READ FAILED: {source_value}"
                )
                return

            new_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            video_fps = float(new_cap.get(cv2.CAP_PROP_FPS))
            if video_fps <= 1.0 or video_fps > 240.0:
                video_fps = 30.0

            if self.is_recording:
                self.stop_recording()

            old_cap = self.cap
            self.cap = new_cap
            if old_cap is not None:
                old_cap.release()

            self.source_type = "video"
            self.source_path = str(Path(source_value).resolve())
            self.source_name = Path(source_value).name
            self.source_fps = video_fps
            self.source_frame_interval = 1.0 / self.source_fps
            self.last_source_frame = preview_frame.copy()

            self.live_on = False
            self.mode = "idle"
            self.reset_runtime_stats()

            self.source_changed_signal.emit(
                "video",
                self.source_name
            )
            self.status_signal.emit(
                f"VIDEO LOADED: {self.source_name}"
            )
            return

        if source_type == "camera":
            camera_index = int(source_value)
            new_cap = cv2.VideoCapture(camera_index)

            if not new_cap.isOpened():
                new_cap.release()
                self.status_signal.emit(
                    f"CAMERA OPEN FAILED: {camera_index}"
                )
                return

            if self.is_recording:
                self.stop_recording()

            old_cap = self.cap
            self.cap = new_cap
            if old_cap is not None:
                old_cap.release()

            camera_fps = float(new_cap.get(cv2.CAP_PROP_FPS))
            if camera_fps <= 1.0 or camera_fps > 240.0:
                camera_fps = 30.0

            self.source_type = "camera"
            self.source_path = None
            self.source_name = f"Camera {camera_index}"
            self.source_fps = camera_fps
            self.source_frame_interval = 1.0 / self.source_fps
            self.last_source_frame = None

            self.live_on = False
            self.mode = "idle"
            self.reset_runtime_stats()

            self.source_changed_signal.emit(
                "camera",
                self.source_name
            )
            self.status_signal.emit(
                f"CAMERA SOURCE READY: {self.source_name}"
            )

    def emit_frame_to_ui(self, frame):
        """Numpy 메모리와 분리된 QImage를 UI로 전달합니다."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        ).copy()
        self.update_frame.emit(qimg)

    def finish_video_playback(self):
        """영상 끝에 도달하면 처음으로 되감고 Live 상태를 종료합니다."""
        self.live_on = False
        self.mode = "idle"

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, preview_frame = self.cap.read()
            if ret and preview_frame is not None:
                self.last_source_frame = preview_frame.copy()

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.status_signal.emit(
            "VIDEO FINISHED - PRESS START VIDEO TO REPLAY"
        )
        self.playback_finished_signal.emit()

    def draw_processing_time(
            self,
            frame,
            yolo_ms,
            osnet_ms,
            match_ms,
            total_ms
    ):
        smooth_alpha = 0.9

        if self.time_total == 0.0:
            self.time_yolo = yolo_ms
            self.time_osnet = osnet_ms
            self.time_match = match_ms
            self.time_total = total_ms
        else:
            self.time_yolo = (
                    smooth_alpha * self.time_yolo
                    + (1.0 - smooth_alpha) * yolo_ms
            )

            self.time_osnet = (
                    smooth_alpha * self.time_osnet
                    + (1.0 - smooth_alpha) * osnet_ms
            )

            self.time_match = (
                    smooth_alpha * self.time_match
                    + (1.0 - smooth_alpha) * match_ms
            )

            self.time_total = (
                    smooth_alpha * self.time_total
                    + (1.0 - smooth_alpha) * total_ms
            )

        if self.time_total > 0:
            self.processing_fps = 1000.0 / self.time_total
        else:
            self.processing_fps = 0.0

        lines = [
            f"YOLO   : {self.time_yolo:.2f} ms",
            f"OSNet  : {self.time_osnet:.2f} ms",
            f"Match  : {self.time_match:.2f} ms",
            f"Total  : {self.time_total:.2f} ms",
            f"FPS    : {self.processing_fps:.1f}",
        ]

        x = 15
        y = 30
        line_height = 28

        # 글자 배경
        cv2.rectangle(
            frame,
            (5, 5),
            (270, 155),
            (0, 0, 0),
            -1
        )

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    def set_recording_enabled(self, enabled):
        """UI의 녹화 ON/OFF 스위치 상태를 반영합니다."""
        self.recording_enabled = bool(enabled)

        if not self.recording_enabled:
            self.recording_stop_requested = True

    def create_recording_path(self):
        """현재 시간을 파일명으로 사용하고 중복 파일명은 방지합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.recording_dir / f"{timestamp}.mp4"

        index = 1
        while path.exists():
            path = self.recording_dir / f"{timestamp}_{index:02d}.mp4"
            index += 1

        return path

    def start_recording(self, raw_frame):
        """박스, FPS, 마스크가 없는 원본 카메라 프레임 녹화를 시작합니다."""
        if self.is_recording:
            return

        height, width = raw_frame.shape[:2]

        camera_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if camera_fps <= 1.0 or camera_fps > 240.0:
            camera_fps = 30.0

        self.recording_fps = camera_fps
        self.recording_path = self.create_recording_path()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.recording_fps,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            self.video_writer = None
            self.recording_path = None
            self.is_recording = False
            self.status_signal.emit("VIDEO RECORDING START FAILED")
            return

        self.video_writer = writer
        self.is_recording = True
        self.recording_last_time = time.perf_counter()
        self.status_signal.emit(
            f"RECORDING STARTED: {self.recording_path.name}"
        )

    def stop_recording(self):
        """현재 녹화 파일을 닫아 디스크에 정상 저장합니다."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.is_recording and self.recording_path is not None:
            saved_path = self.recording_path
            self.status_signal.emit(f"VIDEO SAVED: {saved_path}")

        self.is_recording = False
        self.recording_path = None
        self.recording_last_time = None

    def write_recording_frame(self, raw_frame):
        """
        모델 처리 때문에 프레임 간격이 길어져도 영상 재생속도가 빨라지지 않도록
        경과 시간만큼 동일 프레임을 보충하여 기록합니다.
        """
        if not self.is_recording or self.video_writer is None:
            return

        now = time.perf_counter()

        if self.recording_last_time is None:
            frame_count = 1
        else:
            elapsed = max(0.0, now - self.recording_last_time)
            frame_count = max(1, int(round(elapsed * self.recording_fps)))

            # 일시적인 긴 정지로 과도한 프레임이 기록되는 것 방지
            max_fill_frames = max(1, int(self.recording_fps * 2.0))
            frame_count = min(frame_count, max_fill_frames)

        for _ in range(frame_count):
            self.video_writer.write(raw_frame)

        self.recording_last_time = now

    def update_recording(self, raw_frame):
        """Live/스위치 상태에 따라 녹화를 시작, 기록 또는 종료합니다."""
        if self.recording_stop_requested:
            if self.is_recording:
                self.stop_recording()
            self.recording_stop_requested = False

        should_record = (
                self.source_type == "camera"
                and self.live_on
                and self.recording_enabled
        )

        if should_record:
            if not self.is_recording:
                self.start_recording(raw_frame)

            self.write_recording_frame(raw_frame)
        elif self.is_recording:
            self.stop_recording()

    def toggle_live(self):
        self.live_on = not self.live_on

        if self.live_on:
            self.mode = "live"

            # 영상이 끝난 상태라면 처음부터 다시 재생
            if self.source_type == "video" and self.cap is not None:
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                if total_frames > 0 and current_frame >= total_frames - 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.reset_runtime_stats()
        else:
            self.mode = "idle"

            if self.is_recording:
                self.recording_stop_requested = True

    def stop(self):
        """창 종료 시 Worker 반복문을 안전하게 종료합니다."""
        self.running = False
        self.recording_stop_requested = True

    def draw_processing_time(
            self,
            frame,
            yolo_ms,
            osnet_ms,
            match_ms,
            total_ms
    ):
        smooth_alpha = 0.9

        if self.time_total == 0.0:
            self.time_yolo = yolo_ms
            self.time_osnet = osnet_ms
            self.time_match = match_ms
            self.time_total = total_ms
        else:
            self.time_yolo = (
                    smooth_alpha * self.time_yolo
                    + (1.0 - smooth_alpha) * yolo_ms
            )

            self.time_osnet = (
                    smooth_alpha * self.time_osnet
                    + (1.0 - smooth_alpha) * osnet_ms
            )

            self.time_match = (
                    smooth_alpha * self.time_match
                    + (1.0 - smooth_alpha) * match_ms
            )

            self.time_total = (
                    smooth_alpha * self.time_total
                    + (1.0 - smooth_alpha) * total_ms
            )

        if self.time_total > 0:
            self.processing_fps = 1000.0 / self.time_total
        else:
            self.processing_fps = 0.0

        lines = [
            f"YOLO   : {self.time_yolo:.2f} ms",
            f"OSNet  : {self.time_osnet:.2f} ms",
            f"Match  : {self.time_match:.2f} ms",
            f"Total  : {self.time_total:.2f} ms",
            f"FPS    : {self.processing_fps:.1f}",
        ]

        x = 15
        y = 30
        line_height = 28

        # 글자 배경
        cv2.rectangle(
            frame,
            (5, 5),
            (270, 155),
            (0, 0, 0),
            -1
        )

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    def set_recording_enabled(self, enabled):
        """UI의 녹화 ON/OFF 스위치 상태를 반영합니다."""
        self.recording_enabled = bool(enabled)

        if not self.recording_enabled:
            self.recording_stop_requested = True

    def create_recording_path(self):
        """현재 시간을 파일명으로 사용하고 중복 파일명은 방지합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.recording_dir / f"{timestamp}.mp4"

        index = 1
        while path.exists():
            path = self.recording_dir / f"{timestamp}_{index:02d}.mp4"
            index += 1

        return path

    def start_recording(self, raw_frame):
        """박스, FPS, 마스크가 없는 원본 카메라 프레임 녹화를 시작합니다."""
        if self.is_recording:
            return

        height, width = raw_frame.shape[:2]

        camera_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if camera_fps <= 1.0 or camera_fps > 240.0:
            camera_fps = 30.0

        self.recording_fps = camera_fps
        self.recording_path = self.create_recording_path()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.recording_fps,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            self.video_writer = None
            self.recording_path = None
            self.is_recording = False
            self.status_signal.emit("VIDEO RECORDING START FAILED")
            return

        self.video_writer = writer
        self.is_recording = True
        self.recording_last_time = time.perf_counter()
        self.status_signal.emit(
            f"RECORDING STARTED: {self.recording_path.name}"
        )

    def stop_recording(self):
        """현재 녹화 파일을 닫아 디스크에 정상 저장합니다."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.is_recording and self.recording_path is not None:
            saved_path = self.recording_path
            self.status_signal.emit(f"VIDEO SAVED: {saved_path}")

        self.is_recording = False
        self.recording_path = None
        self.recording_last_time = None

    def write_recording_frame(self, raw_frame):
        """
        모델 처리 때문에 프레임 간격이 길어져도 영상 재생속도가 빨라지지 않도록
        경과 시간만큼 동일 프레임을 보충하여 기록합니다.
        """
        if not self.is_recording or self.video_writer is None:
            return

        now = time.perf_counter()

        if self.recording_last_time is None:
            frame_count = 1
        else:
            elapsed = max(0.0, now - self.recording_last_time)
            frame_count = max(1, int(round(elapsed * self.recording_fps)))

            # 일시적인 긴 정지로 과도한 프레임이 기록되는 것 방지
            max_fill_frames = max(1, int(self.recording_fps * 2.0))
            frame_count = min(frame_count, max_fill_frames)

        for _ in range(frame_count):
            self.video_writer.write(raw_frame)

        self.recording_last_time = now

    def update_recording(self, raw_frame):
        """Live/스위치 상태에 따라 녹화를 시작, 기록 또는 종료합니다."""
        if self.recording_stop_requested:
            if self.is_recording:
                self.stop_recording()
            self.recording_stop_requested = False

        should_record = self.live_on and self.recording_enabled

        if should_record:
            if not self.is_recording:
                self.start_recording(raw_frame)

            self.write_recording_frame(raw_frame)
        elif self.is_recording:
            self.stop_recording()

    def toggle_live(self):
        self.live_on = not self.live_on

        if self.live_on:
            self.mode = "live"
        else:
            self.mode = "idle"

    def run(self):
        while self.running:
            self.apply_pending_source()

            if self.cap is None or not self.cap.isOpened():
                self.msleep(30)
                continue

            # 불러온 영상은 Start Video 전까지 첫 프레임에서 정지
            if (
                    self.source_type == "video"
                    and self.mode == "idle"
                    and not self.live_on
            ):
                if self.last_source_frame is not None:
                    preview = cv2.resize(
                        self.last_source_frame,
                        (960, 540)
                    )
                    cv2.putText(
                        preview,
                        "VIDEO PAUSED",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    self.emit_frame_to_ui(preview)

                self.msleep(30)
                continue

            loop_start = time.perf_counter()

            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                if self.source_type == "video":
                    self.finish_video_playback()
                else:
                    self.msleep(10)
                continue

            self.last_source_frame = raw_frame.copy()

            # 카메라 원본만 녹화합니다. 불러온 영상은 다시 녹화하지 않습니다.
            if self.source_type == "camera":
                self.update_recording(raw_frame)
            elif self.is_recording:
                self.stop_recording()

            frame = cv2.resize(raw_frame, (960, 540))

            # =========================
            # FPS
            # =========================
            now = time.perf_counter()
            dt = now - self.prev_time
            self.prev_time = now

            if dt > 0:
                current_fps = 1.0 / dt

                if self.fps == 0:
                    self.fps = current_fps
                else:
                    self.fps = self.fps * 0.9 + current_fps * 0.1

            cv2.putText(
                frame,
                f"FPS : {self.fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # =========================
            # MODE ROUTING
            # =========================
            if self.mode == "capture":
                self.capture(frame)

            elif self.mode == "live" and self.live_on:
                self.live(frame)

            self.emit_frame_to_ui(frame)

            # 영상 처리 속도가 원본 FPS보다 빠를 때만 재생 간격을 맞춤
            if self.source_type == "video":
                elapsed = time.perf_counter() - loop_start
                remaining = self.source_frame_interval - elapsed

                if remaining > 0:
                    self.msleep(int(remaining * 1000.0))

        if self.is_recording:
            self.stop_recording()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # =========================
    # FEATURE CAPTURE
    # =========================
    def capture(self, frame):
        # if len(self.buffer) == 0:
        self.status_signal.emit(f"Capturing ID {self.target_id}...  {len(self.buffer)}/100")

        results = self.model(
            frame,
            classes=[0],
            conf=0.5,
            imgsz=416,
            verbose=False,
            retina_masks=True
        )[0]

        if results.boxes is None or results.masks is None:
            return

        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        masks = results.masks.data.cpu().numpy()

        h, w = frame.shape[:2]

        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            feat = self.get_feature(frame, box, mask, h, w)
            if feat is None:
                continue

            self.buffer.append(feat)

        if len(self.buffer) > 100:
            self.memory.add(self.target_id, self.buffer)

            self.status_signal.emit(f"ID {self.target_id} SAVED ✔")

            self.mode = "idle"

    def clear(self, id):
        self.buffer = []
        if id in self.memory.base_data.keys():
            del self.memory.base_data[id]
        self.memory.real_time_data[id] = []


    # =========================
    # LIVE MODE
    # =========================
    def live(self, frame):
        total_start = time.perf_counter()

        # =========================
        # 1. YOLO 처리 시간
        # =========================
        yolo_start = time.perf_counter()

        results = self.model(
            frame,
            classes=[0],
            conf=0.5,
            imgsz=416,
            verbose=False,
            retina_masks=True
        )[0]

        yolo_ms = (time.perf_counter() - yolo_start) * 1000.0

        if results.boxes is None or results.masks is None:
            self.draw_processing_time(
                frame=frame,
                yolo_ms=yolo_ms,
                osnet_ms=0.0,
                match_ms=0.0,
                total_ms=(time.perf_counter() - total_start) * 1000.0
            )
            return


        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        masks = results.masks.data.cpu().numpy()

        h, w = frame.shape[:2]

        # =========================
        # 2. 유효한 Crop과 Box 수집
        # =========================
        reid_crops = []
        valid_boxes = []

        crop_start = time.perf_counter()

        for box, mask in zip(boxes, masks):
            crop = self.get_reid_crop(
                frame=frame,
                box=box,
                mask=mask,
                h=h,
                w=w
            )

            if crop is None:
                continue

            reid_crops.append(crop)
            valid_boxes.append(box)

        crop_ms = (time.perf_counter() - crop_start) * 1000.0

        # =========================
        # 3. OSNet Batch 추론
        # =========================
        if reid_crops:
            osnet_start = time.perf_counter()
            features = self.reid.extract_batch(reid_crops)
            osnet_inference_ms = (time.perf_counter() - osnet_start) * 1000.0
        else:
            features = np.empty(
                (0, 512),
                dtype=np.float32
            )
            osnet_inference_ms = 0.0


        total_osnet_ms = crop_ms + osnet_inference_ms

        # =========================
        # 4. Feature Matching
        # =========================
        total_match_ms = 0.0


        re_id = {} # 중복 탐지 방지용

        for box, feature in zip(valid_boxes, features):
            match_start = time.perf_counter()

            gid, score = self.memory.match(feature)


            match_ms = (time.perf_counter() - match_start) * 1000.0
            total_match_ms += match_ms

            if gid is None:
                color = (0, 0, 255)
                label = ( f"UNKNOWN!!! ")
                self.draw_bounding_box(color, label, frame, box)
            else:
                if gid in re_id:
                    if re_id[gid][0] < score:
                        color = (255, 255, 255)
                        label = (f"Mismatching!! ")
                        self.draw_bounding_box(color, label, frame, re_id[gid][1])

                        re_id[gid] = [score, box]
                    else:
                        color = (255, 255, 255)
                        label = (f"Mismatching!! ")
                        self.draw_bounding_box(color, label, frame, box)
                else:
                    re_id[gid] = [score, box]

        id_list = []
        for gid, value in re_id.items():
            score, box = value
            color = self.id_color[gid]
            label = (f"GID:{gid} Score:{score:.3f}. ")
            self.draw_bounding_box(color, label, frame, box)

            id_list.append(gid)

        # 실시간 저장 feature정리
        self.memory.subtract_real_time_cont(id_list)
        self.memory.subtract_unknown_data_cont()


        total_ms = (time.perf_counter() - total_start) * 1000.0

        self.draw_processing_time(
            frame=frame,
            yolo_ms=yolo_ms,
            osnet_ms=total_osnet_ms,
            match_ms=total_match_ms,
            total_ms=total_ms
        )

    def draw_bounding_box(self, color, label, frame, box):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    def get_feature(self, frame, box, mask, h, w):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        # mask crop
        crop_mask = mask[y1:y2, x1:x2]

        # mask 이진화
        crop_mask = (crop_mask > self.mask_thres).astype(np.uint8)

        # crop 영역 안에서 배경 흰색 처리
        output_crop = self.apply_background_white(
            image=crop,
            person_mask=crop_mask
        )

        feat = self.reid.extract(output_crop)

        return feat

    def get_reid_crop(self, frame, box, mask, h, w):
        x1, y1, x2, y2 = map(int, box)

        # 좌표 범위 제한
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            return None

        # retina_masks=True이면 보통 frame과 같은 크기지만
        # 크기가 다른 경우 원본 크기로 보정
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

        crop_mask = mask[y1:y2, x1:x2]

        if crop_mask is None or crop_mask.size == 0:
            return None

        if crop_mask.shape[:2] != crop.shape[:2]:
            crop_mask = cv2.resize(
                crop_mask,
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        crop_mask = (
                crop_mask > self.mask_thres
        ).astype(np.uint8)

        output_crop = self.apply_background_white(
            image=crop,
            person_mask=crop_mask
        )

        return output_crop


    def apply_background_white(self, image, person_mask):
        """
        사람 영역은 원본 유지, 배경은 흰색으로 처리
        """
        alpha = self.smooth_mask(person_mask, blur_size=21)
        alpha_3ch = np.repeat(alpha[:, :, None], 3, axis=2)

        white_bg = np.ones_like(image, dtype=np.uint8) * 255

        output = image.astype(np.float32) * alpha_3ch + white_bg.astype(np.float32) * (1 - alpha_3ch)
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output


    def smooth_mask(self, binary_mask, blur_size=21):
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

    # def real_time_seg_view(self, frame, masks, h, w):
    #     full_person_mask = np.zeros((h, w), dtype=np.uint8)
    #
    #     for mask in masks:
    #         binary_mask = (mask > self.mask_thres).astype(np.uint8)
    #         full_person_mask = np.maximum(full_person_mask, binary_mask)
    #
    #     self.output_full = self.apply_background_white(
    #         image=frame,
    #         person_mask=full_person_mask
    #     )



# =========================
# MAIN UI
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ReID System")

        self.memory = GlobalMemory()
        self.worker = Worker(self.memory)

        self.selected_id = None
        self.current_source_type = "camera"
        self.current_source_name = "Camera 0"

        try:
            self.feature_dir = Path(__file__).resolve().parent
        except NameError:
            self.feature_dir = Path.cwd()

        # =========================
        # UI WIDGETS
        # =========================
        self.video = QLabel()
        self.video.setFixedSize(960, 540)

        self.status = QLabel("SELECT ID")
        self.status.setStyleSheet("color: blue; font-size: 16px;")

        self.source_label = QLabel("Source: Camera 0")
        self.source_label.setStyleSheet(
            "font-size: 14px; font-weight: bold;"
        )

        # =========================
        # ID BUTTONS (0~7)
        # =========================
        self.id_labels = {}

        id_layout = QHBoxLayout()

        for i in range(8):
            btn = QPushButton(f"ID {i}")
            btn.clicked.connect(lambda _, x=i: self.select_id(x))

            label = QLabel("❌")
            label.setAlignment(Qt.AlignCenter)

            clear_btn = QPushButton(f"Clear ID {i}")
            clear_btn.clicked.connect(lambda _, x=i: self.clear(x))

            save_feature_btn = QPushButton("Save Feature")
            save_feature_btn.clicked.connect(
                lambda _, x=i: self.save_feature(x)
            )

            load_feature_btn = QPushButton("Load Feature")
            load_feature_btn.clicked.connect(
                lambda _, x=i: self.load_feature(x)
            )

            vbox = QVBoxLayout()
            vbox.addWidget(btn)
            vbox.addWidget(label)
            vbox.addWidget(clear_btn)
            vbox.addWidget(save_feature_btn)
            vbox.addWidget(load_feature_btn)

            box = QWidget()
            box.setLayout(vbox)

            id_layout.addWidget(box)
            self.id_labels[i] = label

        # =========================
        # SOURCE BUTTONS
        # =========================
        self.btn_load_video = QPushButton("Load Recorded Video")
        self.btn_camera = QPushButton("Use Camera")

        self.btn_load_video.clicked.connect(self.load_recorded_video)
        self.btn_camera.clicked.connect(self.use_camera_source)

        source_button_layout = QHBoxLayout()
        source_button_layout.addWidget(self.btn_load_video)
        source_button_layout.addWidget(self.btn_camera)

        # =========================
        # CONTROL BUTTONS
        # =========================
        self.btn_capture = QPushButton("Feature Capture")
        self.btn_live = QPushButton("Start Live")

        # 녹화 ON/OFF 스위치
        self.btn_record = QPushButton("Recording ON")
        self.btn_record.setCheckable(True)
        self.btn_record.setChecked(True)
        self.update_record_button_style(True)

        self.btn_capture.clicked.connect(self.start_capture)
        self.btn_live.clicked.connect(self.toggle_live)
        self.btn_record.toggled.connect(self.toggle_recording)

        # =========================
        # LAYOUT
        # =========================
        layout = QVBoxLayout()
        layout.addWidget(self.video)
        layout.addWidget(self.source_label)
        layout.addWidget(self.status)
        layout.addLayout(source_button_layout)
        layout.addLayout(id_layout)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_live)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # signals
        self.worker.update_frame.connect(self.update_frame)
        self.worker.status_signal.connect(self.update_status)
        self.worker.source_changed_signal.connect(
            self.on_source_changed
        )
        self.worker.playback_finished_signal.connect(
            self.on_video_finished
        )

        self.worker.start()

    # =========================
    # SOURCE CONTROL
    # =========================
    def load_recorded_video(self):
        start_dir = str(Path.cwd())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "녹화 영상 선택",
            start_dir,
            (
                "Video Files "
                "(*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;"
                "All Files (*)"
            )
        )

        if not file_path:
            return

        self.status.setText("LOADING VIDEO...")
        self.worker.request_video_source(file_path)

    def use_camera_source(self):
        self.status.setText("OPENING CAMERA...")
        self.worker.request_camera_source(0)

    def on_source_changed(self, source_type, source_name):
        self.current_source_type = source_type
        self.current_source_name = source_name
        self.source_label.setText(f"Source: {source_name}")

        if source_type == "video":
            self.btn_live.setText("Start Video")
            self.btn_record.setEnabled(False)
            self.btn_record.setToolTip(
                "불러온 영상 재생 중에는 원본 녹화를 사용하지 않습니다."
            )
        else:
            self.btn_live.setText("Start Live")
            self.btn_record.setEnabled(True)
            self.btn_record.setToolTip("")

    def on_video_finished(self):
        self.btn_live.setText("Start Video")
        self.status.setText(
            "VIDEO FINISHED - START VIDEO를 누르면 처음부터 재생됩니다."
        )

    # =========================
    # ID SELECT
    # =========================
    def select_id(self, gid):
        self.selected_id = gid
        self.status.setText(f"Selected ID: {gid}")

    # =========================
    # FEATURE CAPTURE
    # =========================
    def start_capture(self):
        if self.selected_id is None:
            self.status.setText("Select ID first!")
            return

        self.worker.target_id = self.selected_id
        self.worker.live_on = False
        self.worker.mode = "capture"
        self.clear(self.selected_id)
        self.status.setText(
            f"Capturing ID {self.selected_id}..."
        )

    # =========================
    # FEATURE CLEAR
    # =========================
    def clear(self, id):
        self.worker.clear(id)
        self.id_labels[id].setText("❌")
        self.status.setText(f"ID {id} FEATURE CLEARED")
        print("FEATURE CLEAR!!")

    # =========================
    # FEATURE FILE SAVE / LOAD
    # =========================
    def save_feature(self, gid):
        features = self.memory.get_base_features_copy(gid)

        if features is None or features.shape[0] == 0:
            self.status.setText(
                f"ID {gid}: 저장할 Feature가 없습니다."
            )
            QMessageBox.warning(
                self,
                "Feature 저장",
                f"ID {gid}에 저장된 Feature가 없습니다."
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = self.feature_dir / (
            f"reid_feature_id_{gid}_{timestamp}.npz"
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"ID {gid} Feature 저장",
            str(default_path),
            "ReID Feature Files (*.npz);;All Files (*)"
        )

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
            self.status.setText(f"FEATURE SAVE FAILED: {error}")
            QMessageBox.critical(
                self,
                "Feature 저장 실패",
                str(error)
            )
            return

        self.feature_dir = save_path.parent
        self.status.setText(
            f"ID {gid} FEATURE SAVED: "
            f"{save_path.name} ({features.shape[0]} vectors)"
        )

    def load_feature(self, gid):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"ID {gid}에 불러올 Feature 선택",
            str(self.feature_dir),
            "ReID Feature Files (*.npz);;All Files (*)"
        )

        if not file_path:
            return

        load_path = Path(file_path)

        try:
            with np.load(str(load_path), allow_pickle=False) as data:
                if "features" not in data.files:
                    raise ValueError(
                        "선택한 파일에 'features' 데이터가 없습니다."
                    )

                features = np.asarray(
                    data["features"],
                    dtype=np.float32
                )

                saved_gid = None
                if "gid" in data.files:
                    saved_gid_array = np.asarray(data["gid"]).reshape(-1)
                    if saved_gid_array.size > 0:
                        saved_gid = int(saved_gid_array[0])

            feature_count = self.memory.set_base_features(
                gid,
                features
            )
        except Exception as error:
            self.status.setText(f"FEATURE LOAD FAILED: {error}")
            QMessageBox.critical(
                self,
                "Feature 불러오기 실패",
                str(error)
            )
            return

        # 캡처 도중 남아 있던 임시 Feature 제거
        self.worker.buffer = []
        self.feature_dir = load_path.parent
        self.id_labels[gid].setText(f"✔ ({feature_count})")

        source_id_text = (
            "unknown"
            if saved_gid is None
            else str(saved_gid)
        )
        self.status.setText(
            f"FEATURE LOADED: file ID {source_id_text} -> "
            f"ID {gid}, {feature_count} vectors"
        )

    # =========================
    # RECORDING TOGGLE
    # =========================
    def toggle_recording(self, checked):
        self.worker.set_recording_enabled(checked)
        self.update_record_button_style(checked)

        if checked:
            self.btn_record.setText("Recording ON")
            if (
                    self.worker.live_on
                    and self.current_source_type == "camera"
            ):
                self.status.setText("RECORDING ENABLED")
        else:
            self.btn_record.setText("Recording OFF")
            if self.worker.live_on:
                self.status.setText(
                    "RECORDING DISABLED - SAVING VIDEO"
                )

    def update_record_button_style(self, checked):
        if checked:
            self.btn_record.setStyleSheet(
                "background-color: #2e7d32; "
                "color: white; font-weight: bold;"
            )
        else:
            self.btn_record.setStyleSheet(
                "background-color: #616161; "
                "color: white; font-weight: bold;"
            )

    # =========================
    # LIVE / VIDEO PLAYBACK TOGGLE
    # =========================
    def toggle_live(self):
        self.worker.toggle_live()

        if self.worker.live_on:
            if self.current_source_type == "video":
                self.btn_live.setText("Stop Video")
                self.status.setText(
                    f"VIDEO PROCESSING STARTED: {self.current_source_name}"
                )
            else:
                self.btn_live.setText("Stop Live")

                if self.btn_record.isChecked():
                    self.status.setText(
                        "LIVE MODE STARTED - RECORDING"
                    )
                else:
                    self.status.setText(
                        "LIVE MODE STARTED - RECORDING OFF"
                    )
        else:
            if self.current_source_type == "video":
                self.btn_live.setText("Start Video")
                self.status.setText("VIDEO PAUSED")
            else:
                self.btn_live.setText("Start Live")

                if self.worker.is_recording:
                    self.status.setText(
                        "LIVE MODE STOPPED - SAVING VIDEO"
                    )
                else:
                    self.status.setText("LIVE MODE STOPPED")

    # =========================
    # STATUS UPDATE
    # =========================
    def update_status(self, text):
        self.status.setText(text)

        # Feature 저장 메시지만 ID 체크 표시 처리
        if text.startswith("ID ") and "SAVED" in text:
            gid = self.worker.target_id
            if gid in self.id_labels:
                feature_count = self.memory.get_feature_count(gid)
                self.id_labels[gid].setText(
                    f"✔ ({feature_count})"
                )

    # =========================
    # FRAME UPDATE
    # =========================
    def update_frame(self, qimg):
        self.video.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """창 종료 시 영상 저장과 VideoCapture를 안전하게 정리합니다."""
        self.worker.stop()
        self.worker.wait()
        event.accept()

# =========================
# RUN
# =========================
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())