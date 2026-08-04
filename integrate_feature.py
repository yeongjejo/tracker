
from pathlib import Path
import numpy as np
from datetime import datetime


id_feature = [
    np.empty((0, 512), dtype=np.float32)
    for _ in range(8)
]


def load_feature(file_path):
    try:
        with np.load(str(file_path), allow_pickle=False) as data:
            if "features" not in data.files:
                raise ValueError(
                    "선택한 파일에 'features' 데이터가 없습니다."
                )

            features = np.asarray(
                data["features"],
                dtype=np.float32
            )

            id_feature[data["gid"][0]] = np.concatenate(
                [id_feature[data["gid"][0]], features],
                axis=0
            )
    except Exception as error:
        print(error)
        return


def save_feature(features, gid, file_path):
    if features is None or features.shape[0] == 0:
        print(f"ID {gid}에 저장된 Feature가 없습니다.")
        return

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
        print(error)
        return



if __name__ == '__main__':
    folder_path = Path("C:/Users/USER/Desktop/tracking/testfeature")
    feature_files = sorted(folder_path.glob("*.npz"))

    for path in folder_path.glob("*.npz"):
        load_feature(path)

    save_file_path = "C:/Users/USER/Desktop/tracking/testfeature/test_"
    for id, feature in enumerate(id_feature):
        save_feature(feature, id, save_file_path+str(id)+".npz")
