from pathlib import Path

import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor


MODEL_NAME = "osnet_x1_0"
OUTPUT_PATH = "osnet_x1_0.onnx"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = FeatureExtractor(
        model_name=MODEL_NAME,
        device=device
    )

    model = extractor.model
    model.eval()
    model.to(device)

    # Torchreid OSNet 기본 입력: NCHW = [1, 3, 256, 128]
    dummy_input = torch.randn(
        1,
        3,
        256,
        128,
        device=device
    )

    output_path = Path(OUTPUT_PATH)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["features"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "features": {0: "batch"}
        }
    )

    print(f"ONNX saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()