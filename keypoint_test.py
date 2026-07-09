from ultralytics import YOLO
import cv2

# YOLO pose 모델 로드
# yolo26n-pose.pt 또는 yolo11n-pose.pt / yolov8n-pose.pt 사용 가능
model = YOLO("yolo26n-pose.pt")

img_path = "player1.png"

results = model(img_path)

for result in results:
    # 원본 이미지
    img = result.orig_img.copy()

    # keypoints 객체
    keypoints = result.keypoints

    if keypoints is None:
        continue

    # xy: [사람 수, 키포인트 수, 2]
    # conf: [사람 수, 키포인트 수]
    xy = keypoints.xy.cpu().numpy()
    conf = keypoints.conf.cpu().numpy()

    for person_idx, person_kpts in enumerate(xy):
        print(f"\nPerson {person_idx}")

        for kpt_idx, (x, y) in enumerate(person_kpts):
            score = conf[person_idx][kpt_idx]

            if score < 0.5:
                continue

            print(f"keypoint {kpt_idx}: x={x:.1f}, y={y:.1f}, conf={score:.2f}")

            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imshow("YOLO Pose Keypoints", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()