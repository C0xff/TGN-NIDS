"""Chạy thử notebook huấn luyện trên máy cục bộ trước khi đẩy lên Kaggle.

Mục đích duy nhất là bắt lỗi chạy: thiếu cột, sai tên khoá, sai kiểu dữ liệu,
sai chữ ký hàm. Không phải để đo. Kịch bản thu nhỏ ba tập và hạ ngân sách vòng
lặp xuống mức tối thiểu nên **mọi con số nó in ra đều vô nghĩa** và tuyệt đối
không được trích vào báo cáo.

Lý do tồn tại: một lần chạy hỏng trên Kaggle tốn vài giờ hạn mức bộ tăng tốc và
chỉ báo lỗi ở ô đầu tiên gặp sự cố. Lượt chạy thử này mất vài phút và đi qua
đúng những đường mã mà bản thật sẽ đi qua. Lần dùng đầu tiên nó bắt được sáu
lỗi, trong đó bốn lỗi chỉ lộ ra sau khi giai đoạn một đã chạy xong.

Cách dùng:

    python scripts/chay_thu_notebook.py notebooks/twoDTS/NB1_train_v2.ipynb

Kịch bản tự dựng một cây thư mục giả lập /kaggle/input bằng liên kết mềm trỏ về
data/raw, rồi thay chuỗi đường dẫn trong mã của từng ô. Trên Windows, việc tạo
liên kết mềm đòi hỏi quyền quản trị hoặc Chế độ nhà phát triển; nếu không bật
được thì dùng tham số --chep để chép tệp thay vì liên kết.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
SANDBOX = PROJECT_ROOT / ".chay_thu"
ROWS_PER_SPLIT = 3000
SHRINK_AFTER_CELL = 18          # ô tham số


def build_sandbox(copy_files: bool) -> tuple[Path, Path]:
    """Dựng cây thư mục thay cho /kaggle/input và /kaggle/working."""
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    input_root = SANDBOX / "input"
    working_root = SANDBOX / "working"
    (input_root / "code").mkdir(parents=True)
    (input_root / "data").mkdir(parents=True)
    working_root.mkdir(parents=True)

    def place(source: Path, target: Path) -> None:
        if copy_files:
            shutil.copy2(source, target)
        else:
            os.symlink(source, target)

    # Thư mục mã nguồn phải mang đúng tên src để mẫu dò của notebook khớp.
    code_target = input_root / "code" / "src"
    if copy_files:
        shutil.copytree(PROJECT_ROOT / "src", code_target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        os.symlink(PROJECT_ROOT / "src", code_target, target_is_directory=True)

    found = sorted(RAW_ROOT.rglob("*.parquet"))
    if not found:
        raise FileNotFoundError(
            f"Không có tệp Parquet nào trong {RAW_ROOT}. Tải dữ liệu thô về "
            f"trước khi chạy thử.")
    for path in found:
        place(path, input_root / "data" / path.name)
    print(f"Đã dựng sân chạy thử: {len(found)} tệp dữ liệu, "
          f"{'chép' if copy_files else 'liên kết mềm'}")
    return input_root, working_root


def shrink(namespace: dict) -> None:
    """Thu nhỏ ba tập và hạ ngân sách vòng lặp xuống mức tối thiểu."""
    for parts in namespace["splits"].values():
        for part_name, frame in parts.items():
            # Lấy mẫu đều theo hop để giữ nguyên thứ tự và giữ đủ hai lớp.
            step = max(1, len(frame) // ROWS_PER_SPLIT)
            parts[part_name] = frame.iloc[::step].head(ROWS_PER_SPLIT).copy()
    namespace["EPOCHS_STAGE_ONE"] = 1
    namespace["EPOCHS_STAGE_TWO"] = 1
    namespace["PATIENCE_STAGE_ONE"] = 1
    namespace["PATIENCE_STAGE_TWO"] = 1
    print(f"[thu nhỏ] mỗi tập còn tối đa {ROWS_PER_SPLIT} dòng, "
          f"mỗi giai đoạn 1 vòng. Số liệu từ đây trở đi không có giá trị đo.")


def main() -> int:
    """Chạy notebook trong vùng thử nghiệm để không ghi đè kết quả đã khóa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="đường dẫn tệp .ipynb cần chạy thử")
    parser.add_argument("--chep", action="store_true",
                        help="chép tệp thay vì tạo liên kết mềm (dùng trên "
                             "Windows khi không bật được Chế độ nhà phát triển)")
    parser.add_argument("--giu-san", action="store_true",
                        help="giữ lại thư mục .chay_thu sau khi chạy xong")
    arguments = parser.parse_args()

    notebook_path = Path(arguments.notebook).resolve()
    if not notebook_path.is_file():
        print(f"Không tìm thấy {notebook_path}")
        return 1

    input_root, working_root = build_sandbox(arguments.chep)
    cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    namespace: dict = {"__name__": "__main__"}
    os.environ.setdefault("MPLBACKEND", "Agg")

    exit_code = 0
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace("/kaggle/input", input_root.as_posix())
        source = source.replace("/kaggle/working", working_root.as_posix())
        # Bỏ dòng lệnh shell của IPython: môi trường cục bộ đã cài sẵn gói.
        source = "\n".join("" if line.lstrip().startswith(("!", "%")) else line
                           for line in source.split("\n"))
        try:
            ast.parse(source)
        except SyntaxError as error:
            print(f"\nÔ {index} sai cú pháp: {error}")
            exit_code = 1
            break
        print(f"\n===== ô {index} =====", flush=True)
        try:
            exec(compile(source, f"cell_{index}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            print(f"\nDỪNG Ở Ô {index}")
            exit_code = 1
            break
        if index == SHRINK_AFTER_CELL:
            shrink(namespace)

    if exit_code == 0:
        print("\nHOÀN TẤT không lỗi. Notebook sẵn sàng để đẩy lên Kaggle.")
    if not arguments.giu_san:
        shutil.rmtree(SANDBOX, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
