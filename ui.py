import random
import sys
import cv2
import numpy as np
import torch
import time

from ultralytics import YOLO
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import os
from pathlib import Path
import onnxruntime as ort

from torchreid.reid.utils.feature_extractor import FeatureExtractor


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
                    # 생성된 TensorRT 엔진을 디스크에 캐시
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": engine_cache_dir,

                    # FP16 TensorRT 엔진 사용
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
        input_tensor = self.preprocess(img_bgr)

        if input_tensor is None:
            return None

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )

        feature = np.asarray(
            outputs[0],
            dtype=np.float32
        ).reshape(-1)

        norm = np.linalg.norm(feature)

        if norm > 1e-12:
            feature = feature / norm

        return feature


# =========================
# MEMORY
# =========================
class GlobalMemory:
    def __init__(self):
        self.base_data = {}  # id -> mean feature
        self.real_time_data = {i: [] for i in range(8)}  # id -> mean feature
        self.unknown_data = {}  # id -> mean feature

    def add(self, gid, features):
        # mean = np.mean(features, axis=0)
        # mean = mean / np.linalg.norm(mean)
        self.base_data[gid] = features

    def match(self, feature):
        best_id = None
        best_score = -1

        base_max_num = 50 # 첫 feature 데이터를 몇개을 평균 낼건지
        real_time_num = 50  # 실시간 feature 데이터를 몇개 사용 할건지

        rt1_score = 0.0
        rt2_score = 0.0
        base_score = 0.0
        for gid, base_features in self.base_data.items():
            # -------------------------
            # 1. base feature 평균
            # -------------------------
            # 50개 랜덤 샘플로 사용할경우
            base_sample_num = min(base_max_num, len(base_features))
            base_sample = random.sample(base_features, base_sample_num)
            # 그냥 100개 전부 사용할 경우
            # base_sample = base_features

            base_mean = np.mean(base_sample, axis=0)
            base_mean = base_mean / np.linalg.norm(base_mean)

            # -------------------------
            # 2. real-time feature 평균
            # -------------------------
            if len(self.real_time_data[gid]) > 0:
                # rt_mean1 = np.mean(self.real_time_data[gid][:], axis=0)
                # rt_mean1 = rt_mean1 / np.linalg.norm(rt_mean1)
                #
                # # 개수와 상관없이 비율로 평균 조절
                # mean = (base_mean * 0.0) + (rt_mean1 * 1.0)
                #
                # base_score = np.dot(feature,base_mean  / np.linalg.norm(base_mean))
                # rt1_score = np.dot(feature,rt_mean1 / np.linalg.norm(rt_mean1))
                mean = self.real_time_data[gid][0]
            else:
                # 실시간 feature가 아직 없으면 base만 사용
                mean = base_mean
                base_score = np.dot(feature, mean / np.linalg.norm(mean))

            mean = mean / np.linalg.norm(mean)

            score = np.dot(feature, mean)

            if score > best_score:
                best_score = score
                best_id = gid

        if best_score > 0.75:
            if len(self.real_time_data[best_id]) == 0:
                self.real_time_data[best_id].append(feature / np.linalg.norm(feature))
            else:
                smooth_alpha = 0.85
                updated = (
                        smooth_alpha * self.real_time_data[best_id][0]
                        + (1.0 - smooth_alpha) * feature
                )
                new_realtime_feature = updated / np.linalg.norm(updated)
                self.real_time_data[best_id][0] = new_realtime_feature

            # self.real_time_data[best_id].append(feature)
            # if len(self.real_time_data[best_id]) > real_time_num:
            #     self.real_time_data[best_id].pop(0)

            return best_id, best_score, np.array([base_score, rt1_score, rt2_score])
        print("UNKNOWN")

        # UNKNOWN
        if len(self.unknown_data.keys()) == 0:
            num = random.randint(0, 9999)
            self.unknown_data[num] = [feature, 1]
        else:
            best_id = None
            best_score = -1
            for gid, value in self.unknown_data.items():
                un_feature = value[0].copy()
                un_feature = un_feature / np.linalg.norm(un_feature)
                score = np.dot(feature, un_feature)

                if score > best_score:
                    best_score = score
                    best_id = gid

            if best_score > 0.75:
                smooth_alpha = 0.85
                updated = (
                        smooth_alpha * self.unknown_data[best_id][0]
                        + (1.0 - smooth_alpha) * feature
                )
                new_realtime_feature = updated / np.linalg.norm(updated)
                self.unknown_data[best_id][0] = new_realtime_feature
                self.unknown_data[best_id][1] += 1
            else:
                num = random.randint(0, 9999)
                if num not in self.unknown_data.keys():
                    self.unknown_data[num] = [feature, 1]




        return None, best_score, np.array([base_score, rt1_score, rt2_score])

# =========================
# WORKER THREAD
# =========================
class Worker(QThread):
    update_frame = Signal(QImage)
    status_signal = Signal(str)

    def __init__(self, memory):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
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

    def toggle_live(self):
        self.live_on = not self.live_on

        if self.live_on:
            self.mode = "live"
        else:
            self.mode = "idle"

    def run(self):
        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            # frame = cv2.resize(frame, (960, 540))

            # =========================
            # FPS
            # =========================
            now = time.perf_counter()
            dt = now - self.prev_time
            self.prev_time = now

            if dt > 0:
                current_fps = 1.0 / dt

                # EMA 방식으로 부드럽게
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
            )

            # =========================
            # MODE ROUTING
            # =========================
            if self.mode == "capture":
                self.capture(frame)

            elif self.mode == "live" and self.live_on:
                self.live(frame)

            # =========================
            # UI UPDATE
            # =========================
            # view_frame = cv2.resize(
            #     frame,
            #     (960, 540)
            # )

            # view_frame = self.output_full
            # if view_frame is None:
            #     view_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


            self.update_frame.emit(qimg)

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

        # self.real_time_seg_view(frame, masks, h, w)
        total_osnet_ms = 0.0
        total_match_ms = 0.0
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            # =========================
            # 2. Crop + OSNet 처리 시간
            # =========================
            osnet_start = time.perf_counter()

            feat = self.get_feature(frame, box, mask, h, w)

            osnet_ms = (time.perf_counter() - osnet_start) * 1000.0
            total_osnet_ms += osnet_ms
            if feat is None:
                continue

            # =========================
            # 3. Feature 매칭 시간
            # =========================
            match_start = time.perf_counter()

            gid, score, test_list = self.memory.match(feat)

            match_ms = (time.perf_counter() - match_start) * 1000.0
            total_match_ms += match_ms

            if gid is None:
                color = (0, 0, 255)
                label = f"UNKNOWN Score:{score:.3f}. Base:{test_list[0]:.3f}, rt1:{test_list[1]:.3f}, rt2:{test_list[2]:.3f}"
            else:
                color = (0, 255, 0)
                label = f"GID:{gid} Score:{score:.3f}. Base:{test_list[0]:.3f}, rt1:{test_list[1]:.3f}, rt2:{test_list[2]:.3f}"

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_ms = (time.perf_counter() - total_start) * 1000.0

        self.draw_processing_time(
            frame=frame,
            yolo_ms=yolo_ms,
            osnet_ms=total_osnet_ms,
            match_ms=total_match_ms,
            total_ms=total_ms
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

        # =========================
        # UI WIDGETS
        # =========================
        self.video = QLabel()
        self.video.setFixedSize(960, 540)

        self.status = QLabel("SELECT ID")
        self.status.setStyleSheet("color: blue; font-size: 16px;")

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

            vbox = QVBoxLayout()
            vbox.addWidget(btn)
            vbox.addWidget(label)
            vbox.addWidget(clear_btn)

            box = QWidget()
            box.setLayout(vbox)

            id_layout.addWidget(box)

            self.id_labels[i] = label

        # =========================
        # BUTTONS
        # =========================
        self.btn_capture = QPushButton("Feature Capture")
        self.btn_live = QPushButton("Start Live")

        self.btn_capture.clicked.connect(self.start_capture)
        self.btn_live.clicked.connect(self.toggle_live)

        # =========================
        # LAYOUT
        # =========================
        layout = QVBoxLayout()
        layout.addWidget(self.video)
        layout.addWidget(self.status)
        layout.addLayout(id_layout)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_live)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # signals
        self.worker.update_frame.connect(self.update_frame)
        self.worker.status_signal.connect(self.update_status)

        self.worker.start()

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
        self.worker.mode = "capture"
        self.clear(self.selected_id)
        self.status.setText(f"Capturing ID {self.selected_id}...")


    # =========================
    # FEATURE CLEAR
    # =========================
    def clear(self, id):
        self.worker.clear(id)
        self.id_labels[id].setText("❌")
        print("FEATURE CLEAR!!")


    # =========================
    # LIVE TOGGLE
    # =========================
    def toggle_live(self):
        self.worker.toggle_live()

        if self.worker.live_on:
            self.btn_live.setText("Stop Live")
            self.status.setText("LIVE MODE STARTED")
        else:
            self.btn_live.setText("Start Live")
            self.status.setText("LIVE MODE STOPPED")

    # =========================
    # STATUS UPDATE
    # =========================
    def update_status(self, text):
        self.status.setText(text)

        if "SAVED" in text:
            gid = self.worker.target_id
            self.id_labels[gid].setText("✔")

    # =========================
    # FRAME UPDATE
    # =========================
    def update_frame(self, qimg):
        self.video.setPixmap(QPixmap.fromImage(qimg))


# =========================
# RUN
# =========================
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())










# 출근하면
# osnet 시간 2줄 추가해야하는거하고
# yolo26으로 변경 고려해보고
# 언노운 코드 마무리하고
#