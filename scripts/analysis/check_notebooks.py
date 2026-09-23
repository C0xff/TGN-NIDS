"""Kiểm notebook trước khi đẩy lên Kaggle.

Bắt hai loại lỗi mà chỉ đến lúc chạy trên nền tảng mới lộ ra:

1. Cú pháp Python trong từng ô mã.
2. Vi phạm lược đồ nbformat, kể cả những vi phạm hiện chỉ ở mức cảnh báo.
   Thiếu trường id ở mỗi ô là một ví dụ: lược đồ 4.5 đòi trường này, và bản
   nbformat kế tiếp sẽ đổi cảnh báo đó thành lỗi cứng.

    python scripts/check_notebooks.py

Trả mã 0 nếu mọi notebook đều sạch, mã 1 nếu có vấn đề.
"""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


def check_syntax(path: Path) -> list[str]:
    """Phân tích cú pháp từng ô mã, bỏ qua dòng lệnh shell của IPython."""
    problems = []
    for index, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stripped = "\n".join("" if line.lstrip().startswith(("!", "%")) else line
                             for line in source.split("\n"))
        try:
            ast.parse(stripped)
        except SyntaxError as error:
            problems.append(f"ô {index}: {error}")
    return problems


def check_schema(path: Path) -> list[str]:
    """Đọc và kiểm lược đồ, coi mọi cảnh báo là lỗi."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            nbformat.validate(nbformat.read(str(path), as_version=4))
        except Warning as warning:
            return [f"cảnh báo lược đồ: {warning}"]
        except Exception as error:  # noqa: BLE001 - báo lại nguyên văn
            return [f"{type(error).__name__}: {error}"]
    return []


def main() -> int:
    """Kiểm tra cú pháp và cấu trúc của toàn bộ notebook rồi trả mã trạng thái."""
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    if not notebooks:
        print(f"Không tìm thấy notebook nào trong {NOTEBOOK_ROOT}")
        return 1

    failed = 0
    for path in notebooks:
        problems = check_syntax(path) + check_schema(path)
        relative = path.relative_to(PROJECT_ROOT)
        if problems:
            failed += 1
            print(f"LỖI  {relative}")
            for problem in problems:
                print(f"     {problem}")
        else:
            print(f"sạch {relative}")

    if failed:
        print(f"\n{failed}/{len(notebooks)} notebook có vấn đề. "
              f"Thiếu trường id thì chạy nbformat.validator.normalize().")
        return 1
    print(f"\nCả {len(notebooks)} notebook đều sạch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
