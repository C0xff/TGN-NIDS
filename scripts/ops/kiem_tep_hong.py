"""Dò tệp đọc thiếu byte hoặc hỏng cấu trúc, thường do Google Drive tải dở.

    python scripts/ops/kiem_tep_hong.py              # bỏ qua data/raw
    python scripts/ops/kiem_tep_hong.py --ca-du-lieu # quét cả data/raw
    python scripts/ops/kiem_tep_hong.py --duong-dan models  # chỉ một nhánh

Dự án nằm trên Google Drive dạng phát trực tuyến. Một tệp chưa tải đủ vẫn hiện
trong danh sách thư mục và `os.path.getsize` vẫn trả về dung lượng đầy đủ, nhưng
đọc nội dung thì hụt byte hoặc lỗi giữa chừng. Cùng một cơ chế làm hỏng phần
đuôi của tệp nén, tệp JSON và tệp trọng số mà không báo gì.

Hai phép kiểm chạy trên mỗi tệp:

  1. Đọc hết tệp rồi so số byte đọc được với dung lượng khai báo.
  2. Kiểm cấu trúc theo đuôi tệp. Tệp nén, JSON, notebook, ảnh PNG, PDF,
     Parquet và trọng số PyTorch đều có dấu hiệu nhận biết ở đầu, ở cuối, hoặc
     cả hai, nên cắt cụt là lộ ra ngay.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BO_QUA_THU_MUC = {".git", "__pycache__", ".venv", ".venv-mac", "venv", "env",
                  ".pytest_cache", ".mypy_cache", ".ruff_cache",
                  ".ipynb_checkpoints", "node_modules"}
KHOI = 1 << 22


def doc_het(path: Path) -> tuple[int, int, str | None]:
    """Trả về dung lượng khai báo, số byte đọc được, và lỗi nếu có."""
    khai_bao = path.stat().st_size
    doc_duoc = 0
    try:
        with path.open("rb") as handle:
            while True:
                khoi = handle.read(KHOI)
                if not khoi:
                    break
                doc_duoc += len(khoi)
    except OSError as loi:
        return khai_bao, doc_duoc, "không đọc được: %s" % loi
    return khai_bao, doc_duoc, None


def kiem_cau_truc(path: Path) -> str | None:
    """Kiểm dấu hiệu nhận biết theo đuôi tệp. Trả về mô tả lỗi, hoặc None."""
    duoi = path.suffix.lower()
    try:
        if duoi in {".docx", ".xlsx", ".pptx", ".zip", ".pt", ".pth"}:
            if not zipfile.is_zipfile(path):
                if duoi in {".pt", ".pth"}:
                    return None          # trọng số định dạng cũ, không phải zip
                return "không phải tệp nén hợp lệ"
            with zipfile.ZipFile(path) as z:
                hong = z.testzip()
                if hong:
                    return "mục hỏng bên trong tệp nén: %s" % hong
        elif duoi == ".ipynb":
            noi = json.loads(path.read_text(encoding="utf-8"))
            if "cells" not in noi:
                return "notebook thiếu khoá cells"
        elif duoi == ".json":
            # Tệp cấu hình của trình soạn thảo theo chuẩn JSONC, tức JSON có
            # cho phép ghi chú. Kiểm bằng bộ đọc JSON thuần sẽ báo lỗi giả ở
            # ngay dòng ghi chú đầu tiên, nên chỉ kiểm giải mã UTF-8.
            if ".vscode" in path.parts:
                path.read_text(encoding="utf-8")
            else:
                json.loads(path.read_text(encoding="utf-8"))
        elif duoi in {".md", ".txt", ".py", ".csv", ".toml", ".cfg", ".yml",
                      ".yaml", ".html", ".css", ".js"}:
            path.read_text(encoding="utf-8")
        elif duoi == ".png":
            noi = path.read_bytes()
            if not noi.startswith(b"\x89PNG\r\n\x1a\n"):
                return "thiếu chữ ký PNG ở đầu tệp"
            if not noi.rstrip().endswith(b"IEND\xaeB`\x82"):
                return "thiếu khối IEND ở cuối, tệp bị cắt cụt"
        elif duoi == ".pdf":
            noi = path.read_bytes()
            if not noi.startswith(b"%PDF"):
                return "thiếu chữ ký %PDF ở đầu tệp"
            if b"%%EOF" not in noi[-2048:]:
                return "thiếu dấu %%EOF ở cuối, tệp bị cắt cụt"
        elif duoi == ".parquet":
            with path.open("rb") as handle:
                if handle.read(4) != b"PAR1":
                    return "thiếu chữ ký PAR1 ở đầu tệp"
                handle.seek(-4, 2)
                if handle.read(4) != b"PAR1":
                    return "thiếu chữ ký PAR1 ở cuối, tệp bị cắt cụt"
    except UnicodeDecodeError as loi:
        return "không giải mã được UTF-8 ở byte %d" % loi.start
    except json.JSONDecodeError as loi:
        return "JSON hỏng ở dòng %d cột %d" % (loi.lineno, loi.colno)
    except (OSError, zipfile.BadZipFile) as loi:
        return "%s: %s" % (type(loi).__name__, loi)
    return None


def liet_ke(goc: Path, ca_du_lieu: bool) -> list[Path]:
    ra = []
    for p in goc.rglob("*"):
        if not p.is_file():
            continue
        if any(phan in BO_QUA_THU_MUC for phan in p.parts):
            continue
        if not ca_du_lieu:
            tuong_doi = p.relative_to(PROJECT_ROOT).as_posix()
            if tuong_doi.startswith("data/raw/"):
                continue
        ra.append(p)
    return sorted(ra)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-du-lieu", action="store_true",
                        help="quét cả data/raw, nặng khoảng 13 GB")
    parser.add_argument("--duong-dan", default=".",
                        help="chỉ quét một nhánh, tính từ gốc dự án")
    arguments = parser.parse_args()

    goc = (PROJECT_ROOT / arguments.duong_dan).resolve()
    files = liet_ke(goc, arguments.ca_du_lieu)
    tong = sum(p.stat().st_size for p in files)
    print("Quét %s" % goc)
    print("%d tệp, %.3f GB\n" % (len(files), tong / 1e9))

    cut, rong, hong, da_doc = [], [], [], 0
    for i, p in enumerate(files, 1):
        ten = p.relative_to(PROJECT_ROOT).as_posix()
        khai_bao, doc_duoc, loi = doc_het(p)
        da_doc += doc_duoc
        if loi:
            cut.append((ten, khai_bao, doc_duoc, loi))
        elif khai_bao == 0:
            rong.append(ten)
        elif doc_duoc != khai_bao:
            cut.append((ten, khai_bao, doc_duoc, "đọc hụt byte"))
        else:
            sai = kiem_cau_truc(p)
            if sai:
                hong.append((ten, sai))
        if i % 200 == 0:
            print("  ... %d/%d tệp" % (i, len(files)))

    print()
    print("Đã đọc thật %.3f GB trên %.3f GB khai báo" % (da_doc / 1e9, tong / 1e9))
    print()
    print("Tệp đọc thiếu byte hoặc không đọc được: %d" % len(cut))
    for ten, a, b, loi in cut:
        print("   %s\n      khai báo %s byte, đọc được %s byte, %s"
              % (ten, format(a, ","), format(b, ","), loi))
    print("Tệp hỏng cấu trúc: %d" % len(hong))
    for ten, loi in hong:
        print("   %s\n      %s" % (ten, loi))
    print("Tệp rỗng: %d" % len(rong))
    for ten in rong:
        print("   %s" % ten)

    return 1 if (cut or hong) else 0


if __name__ == "__main__":
    sys.exit(main())
