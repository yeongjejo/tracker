import sys
import time
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QFileDialog,
    QMessageBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap


class CameraCalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OpenCV Chessboard Camera Calibration - PySide6")
        self.resize(1500, 820)

        # =========================
        # Camera
        # =========================
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # =========================
        # Calibration Data
        # =========================
        self.objpoints = []
        self.imgpoints = []

        self.last_found = False
        self.last_corners = None
        self.image_size = None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rms_error = None

        self.map1 = None
        self.map2 = None
        self.map_size = None

        # =========================
        # Auto Capture
        # =========================
        self.auto_capture_enabled = False
        self.auto_capture_target = 30
        self.auto_capture_interval = 0.7
        self.last_auto_capture_time = 0.0

        # =========================
        # UI - Preview Labels
        # =========================
        self.original_title = QLabel("보정 적용 전 원본 화면")
        self.original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.undistorted_title = QLabel("보정 적용 후 화면")
        self.undistorted_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.undistorted_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setStyleSheet("background-color: black; color: white;")
        self.original_label.setText("Original Preview")
        self.original_label.setMinimumSize(600, 450)

        self.undistorted_label = QLabel()
        self.undistorted_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.undistorted_label.setStyleSheet("background-color: black; color: white;")
        self.undistorted_label.setText("Undistorted Preview")
        self.undistorted_label.setMinimumSize(600, 450)

        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_title)
        original_layout.addWidget(self.original_label)

        undistorted_layout = QVBoxLayout()
        undistorted_layout.addWidget(self.undistorted_title)
        undistorted_layout.addWidget(self.undistorted_label)

        preview_layout = QHBoxLayout()
        preview_layout.addLayout(original_layout, stretch=1)
        preview_layout.addLayout(undistorted_layout, stretch=1)

        # =========================
        # UI - Controls
        # =========================
        self.status_label = QLabel("상태: 대기 중")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 14px;")

        self.camera_index_edit = QLineEdit("0")

        # 내부 코너 개수
        # 예: 체스보드 칸이 10 x 7이면 내부 코너는 9 x 6
        self.board_cols_edit = QLineEdit("7")
        self.board_rows_edit = QLineEdit("5")

        # 체스보드 한 칸 실제 크기
        self.square_size_edit = QLineEdit("3.0")

        self.target_count_edit = QLineEdit("30")
        self.capture_interval_edit = QLineEdit("0.7")

        self.start_button = QPushButton("라이브 시작")
        self.stop_button = QPushButton("라이브 정지")
        self.capture_button = QPushButton("자동 코너 캡처 시작")
        self.calibrate_button = QPushButton("보정 진행")
        self.save_button = QPushButton("보정값 저장")
        self.load_button = QPushButton("보정값 불러오기")
        self.clear_button = QPushButton("캡처 데이터 초기화")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button.clicked.connect(self.toggle_auto_capture)
        self.calibrate_button.clicked.connect(self.run_calibration)
        self.save_button.clicked.connect(self.save_calibration)
        self.load_button.clicked.connect(self.load_calibration)
        self.clear_button.clicked.connect(self.clear_samples)

        setting_layout = QGridLayout()
        setting_layout.addWidget(QLabel("Camera Index"), 0, 0)
        setting_layout.addWidget(self.camera_index_edit, 0, 1)

        setting_layout.addWidget(QLabel("내부 코너 가로"), 1, 0)
        setting_layout.addWidget(self.board_cols_edit, 1, 1)

        setting_layout.addWidget(QLabel("내부 코너 세로"), 2, 0)
        setting_layout.addWidget(self.board_rows_edit, 2, 1)

        setting_layout.addWidget(QLabel("한 칸 크기"), 3, 0)
        setting_layout.addWidget(self.square_size_edit, 3, 1)

        setting_layout.addWidget(QLabel("자동 캡처 목표 장수"), 4, 0)
        setting_layout.addWidget(self.target_count_edit, 4, 1)

        setting_layout.addWidget(QLabel("캡처 간격 초"), 5, 0)
        setting_layout.addWidget(self.capture_interval_edit, 5, 1)

        button_layout = QGridLayout()
        button_layout.addWidget(self.start_button, 0, 0)
        button_layout.addWidget(self.stop_button, 0, 1)
        button_layout.addWidget(self.capture_button, 1, 0, 1, 2)
        button_layout.addWidget(self.calibrate_button, 2, 0)
        button_layout.addWidget(self.save_button, 2, 1)
        button_layout.addWidget(self.load_button, 3, 0)
        button_layout.addWidget(self.clear_button, 3, 1)

        control_layout = QVBoxLayout()
        control_layout.addLayout(setting_layout)
        control_layout.addSpacing(10)
        control_layout.addLayout(button_layout)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(preview_layout, stretch=1)
        main_layout.addLayout(control_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # =========================
    # Utility
    # =========================
    def update_status(self, text):
        self.status_label.setText(f"상태: {text}")

    def show_message(self, title, message):
        QMessageBox.information(self, title, message)

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def get_board_size(self, show_error=True):
        try:
            cols = int(self.board_cols_edit.text())
            rows = int(self.board_rows_edit.text())

            if cols <= 0 or rows <= 0:
                raise ValueError

            return cols, rows

        except ValueError:
            if show_error:
                self.show_warning("입력 오류", "내부 코너 가로/세로는 1 이상의 정수여야 합니다.")
            return None

    def get_square_size(self):
        try:
            square_size = float(self.square_size_edit.text())

            if square_size <= 0:
                raise ValueError

            return square_size

        except ValueError:
            self.show_warning("입력 오류", "한 칸 크기는 0보다 큰 숫자여야 합니다.")
            return None

    def get_auto_capture_settings(self):
        try:
            target = int(self.target_count_edit.text())
            interval = float(self.capture_interval_edit.text())

            if target <= 0:
                raise ValueError

            if interval < 0.1:
                raise ValueError

            return target, interval

        except ValueError:
            self.show_warning(
                "입력 오류",
                "자동 캡처 목표 장수는 1 이상 정수, 캡처 간격은 0.1초 이상 숫자로 입력하세요."
            )
            return None

    def make_object_points(self):
        board_size = self.get_board_size(show_error=True)
        square_size = self.get_square_size()

        if board_size is None or square_size is None:
            return None

        cols, rows = board_size

        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= square_size

        return objp

    # =========================
    # Camera
    # =========================
    def start_camera(self):
        if self.cap is not None:
            self.stop_camera()

        try:
            camera_index = int(self.camera_index_edit.text())
        except ValueError:
            self.show_warning("오류", "Camera Index는 숫자여야 합니다.")
            return

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # 카메라 해상도 지정이 필요하면 사용
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.cap = None
            self.show_warning("오류", "카메라를 열 수 없습니다.")
            return

        self.timer.start(30)
        self.update_status("라이브 시작")

    def stop_camera(self):
        self.timer.stop()

        self.auto_capture_enabled = False
        self.capture_button.setText("자동 코너 캡처 시작")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.original_label.setText("Original Preview")
        self.undistorted_label.setText("Undistorted Preview")
        self.update_status("라이브 정지")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.update_status("프레임 수신 실패")
            return

        self.image_size = frame.shape[1], frame.shape[0]

        original_display = frame.copy()

        # =========================
        # 체스보드 감지
        # =========================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        board_size = self.get_board_size(show_error=False)

        if board_size is not None:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK
            )

            found, corners = cv2.findChessboardCorners(gray, board_size, flags)
            self.last_found = found

            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )

                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

                self.last_corners = corners2

                cv2.drawChessboardCorners(original_display, board_size, corners2, found)

                if self.auto_capture_enabled:
                    self.auto_capture_corners()

                cv2.putText(
                    original_display,
                    "Chessboard: FOUND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            else:
                self.last_corners = None

                cv2.putText(
                    original_display,
                    "Chessboard: NOT FOUND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

        # =========================
        # 원본 화면 표시용 텍스트
        # =========================
        cv2.putText(
            original_display,
            f"Samples: {len(self.objpoints)} / {self.auto_capture_target}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        if self.auto_capture_enabled:
            cv2.putText(
                original_display,
                "AUTO CAPTURE: ON",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

        cv2.putText(
            original_display,
            "BEFORE CALIBRATION",
            (20, original_display.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # =========================
        # 보정 후 화면 생성
        # =========================
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            undistorted_display = self.undistort_frame(frame.copy())

            cv2.putText(
                undistorted_display,
                "AFTER CALIBRATION",
                (20, undistorted_display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        else:
            undistorted_display = np.zeros_like(frame)

            cv2.putText(
                undistorted_display,
                "No Calibration Data",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            cv2.putText(
                undistorted_display,
                "Run calibration or load calibration file",
                (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        self.show_frame(self.original_label, original_display)
        self.show_frame(self.undistorted_label, undistorted_display)

    def show_frame(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(pixmap)

    # =========================
    # Auto Capture
    # =========================
    def toggle_auto_capture(self):
        if self.auto_capture_enabled:
            self.auto_capture_enabled = False
            self.capture_button.setText("자동 코너 캡처 시작")
            self.update_status(f"자동 캡처 중지 / 현재 {len(self.objpoints)}장")
            return

        if self.cap is None:
            self.show_warning("자동 캡처 불가", "먼저 라이브를 시작하세요.")
            return

        settings = self.get_auto_capture_settings()
        if settings is None:
            return

        self.auto_capture_target, self.auto_capture_interval = settings

        if len(self.objpoints) >= self.auto_capture_target:
            self.show_warning(
                "자동 캡처 불가",
                f"이미 {len(self.objpoints)}장이 저장되어 있습니다.\n"
                f"캡처 데이터 초기화 후 다시 시작하세요."
            )
            return

        self.auto_capture_enabled = True
        self.last_auto_capture_time = 0.0
        self.capture_button.setText("자동 코너 캡처 중지")

        self.update_status(
            f"자동 캡처 시작 / 목표 {self.auto_capture_target}장 / "
            f"간격 {self.auto_capture_interval}초"
        )

    def auto_capture_corners(self):
        if not self.auto_capture_enabled:
            return

        if self.last_found is False or self.last_corners is None:
            return

        if len(self.objpoints) >= self.auto_capture_target:
            self.finish_auto_capture()
            return

        now = time.time()

        if now - self.last_auto_capture_time < self.auto_capture_interval:
            return

        objp = self.make_object_points()

        if objp is None:
            return

        self.objpoints.append(objp.copy())
        self.imgpoints.append(self.last_corners.copy())

        self.last_auto_capture_time = now

        count = len(self.objpoints)

        self.update_status(
            f"자동 캡처 진행 중 / {count} / {self.auto_capture_target}"
        )

        print(f"Auto captured samples: {count}")

        if count >= self.auto_capture_target:
            self.finish_auto_capture()

    def finish_auto_capture(self):
        self.auto_capture_enabled = False
        self.capture_button.setText("자동 코너 캡처 시작")

        count = len(self.objpoints)

        self.update_status(f"자동 캡처 완료 / 총 {count}장")
        self.show_message(
            "자동 캡처 완료",
            f"총 {count}장의 체스보드 코너를 캡처했습니다.\n\n"
            f"이제 '보정 진행' 버튼을 누르세요."
        )

    def clear_samples(self):
        self.auto_capture_enabled = False
        self.capture_button.setText("자동 코너 캡처 시작")

        self.objpoints.clear()
        self.imgpoints.clear()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rms_error = None

        self.map1 = None
        self.map2 = None
        self.map_size = None

        self.update_status("캡처 데이터 및 보정값 초기화 완료")

    # =========================
    # Calibration
    # =========================
    def run_calibration(self):
        if len(self.objpoints) < 5:
            self.show_warning(
                "보정 불가",
                "최소 5장 이상 캡처한 뒤 보정을 진행하세요.\n"
                "권장: 15~30장 이상"
            )
            return

        if self.image_size is None:
            self.show_warning("보정 불가", "이미지 크기 정보가 없습니다.")
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            None,
            None
        )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rms_error = ret

        self.map1 = None
        self.map2 = None
        self.map_size = None

        reprojection_error = self.calculate_reprojection_error(
            self.objpoints,
            self.imgpoints,
            rvecs,
            tvecs,
            camera_matrix,
            dist_coeffs
        )

        msg = (
            f"보정 완료\n\n"
            f"RMS Error: {ret:.6f}\n"
            f"Mean Reprojection Error: {reprojection_error:.6f}\n\n"
            f"Camera Matrix:\n{camera_matrix}\n\n"
            f"Distortion Coefficients:\n{dist_coeffs}"
        )

        print(msg)
        self.update_status(
            f"카메라 보정 완료 / RMS: {ret:.6f} / Reprojection: {reprojection_error:.6f}"
        )
        self.show_message("보정 완료", msg)

    def calculate_reprojection_error(
        self,
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    ):
        total_error = 0.0

        for i in range(len(objpoints)):
            projected_points, _ = cv2.projectPoints(
                objpoints[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )

            error = cv2.norm(
                imgpoints[i],
                projected_points,
                cv2.NORM_L2
            ) / len(projected_points)

            total_error += error

        mean_error = total_error / len(objpoints)
        return mean_error

    # =========================
    # Undistortion
    # =========================
    def undistort_frame(self, frame):
        h, w = frame.shape[:2]

        if self.map1 is None or self.map2 is None or self.map_size != (w, h):
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (w, h),
                1,
                (w, h)
            )

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_camera_matrix,
                (w, h),
                cv2.CV_16SC2
            )

            self.map_size = (w, h)

        undistorted = cv2.remap(
            frame,
            self.map1,
            self.map2,
            cv2.INTER_LINEAR
        )

        return undistorted

    # =========================
    # Save / Load Calibration
    # =========================
    def save_calibration(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.show_warning("저장 실패", "먼저 보정을 진행해야 합니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "보정값 저장",
            "camera_calibration.npz",
            "Calibration Files (*.npz *.yml *.yaml)"
        )

        if not path:
            return

        lower_path = path.lower()

        if not (
            lower_path.endswith(".npz")
            or lower_path.endswith(".yml")
            or lower_path.endswith(".yaml")
        ):
            path += ".npz"
            lower_path = path.lower()

        if lower_path.endswith(".npz"):
            np.savez(
                path,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_size=np.array(self.image_size),
                rms_error=np.array(self.rms_error)
            )

        elif lower_path.endswith(".yml") or lower_path.endswith(".yaml"):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

            fs.write("camera_matrix", self.camera_matrix)
            fs.write("dist_coeffs", self.dist_coeffs)

            if self.image_size is not None:
                fs.write("image_width", int(self.image_size[0]))
                fs.write("image_height", int(self.image_size[1]))

            if self.rms_error is not None:
                fs.write("rms_error", float(self.rms_error))

            fs.release()

        self.update_status(f"보정값 저장 완료: {path}")
        self.show_message("저장 완료", f"보정값을 저장했습니다.\n\n{path}")

    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "보정값 불러오기",
            "",
            "Calibration Files (*.npz *.yml *.yaml)"
        )

        if not path:
            return

        lower_path = path.lower()

        if lower_path.endswith(".npz"):
            data = np.load(path)

            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

            if "image_size" in data:
                image_size = data["image_size"]
                self.image_size = tuple(image_size.tolist())

            if "rms_error" in data:
                self.rms_error = float(data["rms_error"])

        elif lower_path.endswith(".yml") or lower_path.endswith(".yaml"):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

            self.camera_matrix = fs.getNode("camera_matrix").mat()
            self.dist_coeffs = fs.getNode("dist_coeffs").mat()

            image_width = fs.getNode("image_width").real()
            image_height = fs.getNode("image_height").real()

            if image_width > 0 and image_height > 0:
                self.image_size = (int(image_width), int(image_height))

            rms = fs.getNode("rms_error").real()
            if rms > 0:
                self.rms_error = rms

            fs.release()

        else:
            self.show_warning("불러오기 실패", "지원하지 않는 파일 형식입니다.")
            return

        if self.camera_matrix is None or self.dist_coeffs is None:
            self.show_warning("불러오기 실패", "보정값을 읽지 못했습니다.")
            return

        self.map1 = None
        self.map2 = None
        self.map_size = None

        msg = (
            f"보정값 불러오기 완료\n\n"
            f"Camera Matrix:\n{self.camera_matrix}\n\n"
            f"Distortion Coefficients:\n{self.dist_coeffs}"
        )

        print(msg)
        self.update_status(f"보정값 불러오기 완료: {path}")
        self.show_message("불러오기 완료", msg)

    # =========================
    # Close
    # =========================
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = CameraCalibrationGUI()
    window.show()

    sys.exit(app.exec())