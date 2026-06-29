import cv2
import numpy as np
import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor

class GlobalMemory:
    def __init__(self, threshold=0.75, max_feat=50):
        self.people = {}  # g_id -> feature list
        self.threshold = threshold
        self.max_feat = max_feat
        self.next_id = 0

    def _mean(self, feats):
        f = np.mean(feats, axis=0)
        return f / np.linalg.norm(f)

    def add_new(self, feature):
        gid = self.next_id
        self.next_id += 1

        self.people[gid] = [feature]
        return gid

    def update(self, gid, feature):
        self.people[gid].append(feature)

        if len(self.people[gid]) > self.max_feat:
            self.people[gid] = self.people[gid][-self.max_feat:]

    def match(self, feature):
        best_id = None
        best_score = -1

        for gid, feats in self.people.items():
            ref = self._mean(feats)
            score = np.dot(feature, ref)

            if score > best_score:
                best_score = score
                best_id = gid

        if best_score > self.threshold:
            return best_id, best_score

        return None, best_score


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


class FeatureBank:
    def __init__(self, max_history=30):
        self.bank = {}  # id -> list(features)
        self.max_history = max_history

    def update(self, track_id, feature):
        if track_id not in self.bank:
            self.bank[track_id] = []

        self.bank[track_id].append(feature)

        if len(self.bank[track_id]) > self.max_history:
            self.bank[track_id] = self.bank[track_id][-self.max_history:]

    def get_mean(self, track_id):
        feats = self.bank.get(track_id, None)

        if feats is None or len(feats) == 0:
            return None

        mean_feat = np.mean(feats, axis=0)
        mean_feat = mean_feat / np.linalg.norm(mean_feat)

        return mean_feat

    def remove(self, track_id):
        if track_id in self.bank:
            del self.bank[track_id]


def cosine_sim(a, b):
    if a is None or b is None:
        return -1

    return np.dot(a, b)

from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):
        results = self.model(frame, classes=[0], verbose=False)

        if len(results) == 0:
            return None

        return results[0]

import cv2
# =========================
# 초기화
# =========================

detector = Detector()
reid = ReIDModel()
memory = GlobalMemory(threshold=0.75)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('video.mp4')
cv2.namedWindow("GLOBAL REID", cv2.WINDOW_NORMAL)
cv2.moveWindow("GLOBAL REID", 800, 200)
while True:
    ret, frame = cap.read()
    if not ret:
        break


    # frame = cv2.resize(frame, (1280, 720))  # or (960, 540)
    results = detector.detect(frame)
    # =========================

    if results is None:
        cv2.imshow("GLOBAL REID", frame)
        continue

    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        cv2.imshow("GLOBAL REID", frame)
        continue

    for i in range(len(boxes)):

        box = boxes.xyxy[i]
        x1, y1, x2, y2 = map(int, box)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        feature = reid.extract(crop)
        if feature is None:
            continue

        gid, score = memory.match(feature)

        if gid is None:
            gid = memory.add_new(feature)
            color = (255, 0, 0)
        else:
            memory.update(gid, feature)
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"GID:{gid} S:{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("GLOBAL REID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()