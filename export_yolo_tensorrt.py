from ultralytics import YOLO


def main():
    model = YOLO("yolo26n-seg.pt")

    output_path = model.export(
        format="engine",
        imgsz=416,
        quantize=16,
        device=0,
        batch=1,
        dynamic=False,
        workspace=4,
    )

    print(f"TensorRT engine saved: {output_path}")


if __name__ == "__main__":
    main()
