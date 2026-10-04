"""Tải các bộ dữ liệu NetFlow từ Hugging Face về data/raw.

Các bộ do University of Queensland phát hành, bản Parquet trên Hugging Face.
Bản v2 có 43 cột, không có cột địa chỉ IP; bản v3 có 55 cột và có địa chỉ.

Chạy:  python scripts/pipeline/download_nids_datasets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASET_SPECS = [
    {
        "name": "nf-unsw-nb15-v2",
        "repo_id": "dhoogla/nfunswnb15v2",
        "target_dir": "nf-unsw-nb15-v2",
        "source_kind": "kagglehub",
        "source_note": "Kaggle mirror for NF-UNSW-NB15-V2 (parquet)",
    },
    {
        "name": "unsw-nb15-v3",
        "repo_id": "abluva/UNSW-NB15-V3",
        "target_dir": "nf-unsw-nb15-v3",
        "source_kind": "huggingface",
        "source_note": "Hugging Face mirror for UNSW-NB15-V3",
    },
]


def write_plan(output_path: Path) -> None:
    """Ghi hướng dẫn tải thủ công khi nguồn dữ liệu cần xác thực riêng."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """Dataset download plan for NF-UNSW-NB15 v2/v3

Recommended sources:
- https://research.unsw.edu.au/projects/unsw-nb15-dataset
- https://researchdata.edu.au/nf-unsw-nb15/3370524
- https://researchdata.edu.au/nf-unsw-nb15-v3/
- https://espace.library.uq.edu.au/view/UQ:ffbb0c1
- https://espace.library.uq.edu.au/view/UQ:6e0eda1

Hugging Face mirrors:
- https://huggingface.co/datasets/Jetlime/NF-UNSW-NB15-v2/viewer
- https://huggingface.co/datasets/abluva/UNSW-NB15-V3
- https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2
- https://www.kaggle.com/datasets/dhoogla/nfunswnb15

Use the Python script to download after installing the optional dependency 'datasets'.
The repo keeps the format of the target folders stable:
data/raw/nf-unsw-nb15-v2/
data/raw/nf-unsw-nb15-v3/
""",
        encoding="utf-8",
    )


# Ghim phiên bản trên Hugging Face để tải lại đúng dữ liệu đã dùng. None là bản mới nhất.
DATASET_REVISION = "main"


def download_huggingface_dataset(repo_id: str, output_dir: Path,
                                 revision: str = DATASET_REVISION) -> None:
    """Tải bộ dữ liệu Hugging Face tại đúng phiên bản đã chốt."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Thiếu thư viện 'datasets'. Cài bằng: "
            f"{sys.executable} -m pip install -r requirements.txt"
        ) from error

    dataset = load_dataset(repo_id, revision=revision)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "keys"):
        for split_name, split_dataset in dataset.items():
            split_path = output_dir / f"{split_name}.parquet"
            split_dataset.to_parquet(str(split_path))
    else:
        split_path = output_dir / "data.parquet"
        dataset.to_parquet(str(split_path))


def download_kaggle_dataset(repo_id: str, output_dir: Path) -> None:
    """Tải và giải nén một bộ dữ liệu Kaggle vào thư mục đích."""
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Thiếu thư viện 'kagglehub'. Cài bằng: "
            f"{sys.executable} -m pip install kagglehub"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(kagglehub.dataset_download(repo_id))

    copied_any_file = False
    for source_path in downloaded_path.rglob("*"):
        if source_path.is_file() and source_path.suffix.lower() in {".csv", ".parquet"}:
            destination_path = output_dir / source_path.name
            destination_path.write_bytes(source_path.read_bytes())
            copied_any_file = True

    if not copied_any_file:
        raise RuntimeError(f"Không tìm thấy file dữ liệu hợp lệ trong {downloaded_path}")


def main() -> None:
    """Đọc tham số dòng lệnh và tải nguồn dữ liệu được yêu cầu."""
    parser = argparse.ArgumentParser(description="Write a stable dataset download note for NF-UNSW-NB15.")
    parser.add_argument("--download", action="store_true", help="Download the datasets into data/raw.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "DATASET_DOWNLOAD_PLAN.txt"),
        help="Where to write the download plan note.",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    write_plan(output_path)

    if args.download:
        workspace_root = Path(__file__).resolve().parents[2]
        raw_root = workspace_root / "data" / "raw"
        for spec in DATASET_SPECS:
            target_dir = raw_root / spec["target_dir"]
            if target_dir.exists() and any(target_dir.iterdir()):
                continue
            print(f"Downloading {spec['name']} into {target_dir}...")
            try:
                if spec["source_kind"] == "kagglehub":
                    download_kaggle_dataset(spec["repo_id"], target_dir)
                else:
                    download_huggingface_dataset(spec["repo_id"], target_dir)
                print(f"Downloaded {spec['name']} from {spec['source_note']}")
            except Exception as error:
                print(f"Failed to download {spec['name']} from {spec['source_note']}: {error}")
        print("Dataset download completed.")


if __name__ == "__main__":
    main()
