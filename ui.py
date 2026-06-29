import sys
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from torchreid.reid.utils.feature_extractor import FeatureExtractor


# =========================
# REID MODEL
# =========================
class ReIDModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name="osnet_ibn_x1_0",
            device=self.device
        )

    def extract(self, img):
        if img is None or img.size == 0:
            return None

        feat = self.extractor(img).cpu().numpy().flatten()
        feat = feat / np.linalg.norm(feat)
        return feat


# =========================
# MEMORY
# =========================
class GlobalMemory:
    def __init__(self):
        self.data = {}  # id -> mean feature

    def add(self, gid, features):
        mean = np.mean(features, axis=0)
        mean = mean / np.linalg.norm(mean)
        self.data[gid] = mean

    def match(self, feature):
        best_id = None
        best_score = -1

        for gid, ref in self.data.items():
            score = np.dot(feature, ref)

            if score > best_score:
                best_score = score
                best_id = gid

        if best_score > 0.75:
            return best_id, best_score

        return None, best_score


# =========================
# WORKER THREAD
# =========================
class Worker(QThread):
    update_frame = Signal(QImage)
    status_signal = Signal(str)

    def __init__(self, memory):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        self.model = YOLO("yolo11n.pt")
        self.reid = ReIDModel()

        self.memory = memory

        self.mode = "idle"      # idle / capture / live
        self.live_on = False

        self.target_id = None
        self.buffer = []

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

            frame = cv2.resize(frame, (960, 540))

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

            self.update_frame.emit(qimg)

    # =========================
    # FEATURE CAPTURE
    # =========================
    def capture(self, frame):

        # if len(self.buffer) == 0:
        self.status_signal.emit(f"Capturing ID {self.target_id}...  {len(self.buffer)}/50")

        results = self.model(frame, classes=[0], verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes.xyxy:

                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                feat = self.reid.extract(crop)
                if feat is not None:
                    self.buffer.append(feat)

        if len(self.buffer) > 50:

            self.memory.add(self.target_id, self.buffer)

            self.status_signal.emit(f"ID {self.target_id} SAVED ✔")

            self.buffer = []
            self.mode = "idle"

    # =========================
    # LIVE MODE
    # =========================
    def live(self, frame):

        results = self.model(frame, classes=[0], verbose=False)[0]

        if results.boxes is None:
            return

        for box in results.boxes.xyxy:

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            feat = self.reid.extract(crop)
            if feat is None:
                continue

            gid, score = self.memory.match(feat)

            if gid is None:
                color = (0, 0, 255)
                label = f"UNKNOWN Score:{score}"
            else:
                color = (0, 255, 0)
                label = f"GID:{gid} Score:{score}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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

            vbox = QVBoxLayout()
            vbox.addWidget(btn)
            vbox.addWidget(label)

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
        self.status.setText(f"Capturing ID {self.selected_id}...")

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