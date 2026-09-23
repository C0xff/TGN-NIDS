"""Sao lưu một bản hoàn thiện của dự án ra ngoài, trước khi bắt đầu viết báo cáo.

Bản sao lưu là mốc để so lại khi một người khác lỡ sửa vào phạm vi khoá. Nó
phải nằm **ngoài** thư mục dự án, vì một lệnh xoá nhầm trong dự án cũng cuốn
theo bản sao nằm bên trong.

    python scripts/backup_ban_hoan_thien.py            # sao lưu
    python scripts/backup_ban_hoan_thien.py --liet-ke  # chỉ liệt kê, không chép

Bỏ qua dữ liệu thô vì tải lại được từ nguồn, và bỏ qua các thư mục nặng không
mang thông tin (môi trường ảo, cache). Trọng số mô hình và kết quả thì **giữ**,
vì đó chính là thứ không tái tạo được nếu mất.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKUP_ROOT = PROJECT_ROOT.parent / "TGN-NIDS-BAN-HOAN-THIEN"

# Thư mục bỏ qua ở mọi cấp.
SKIP_DIRECTORIES = {
    "__pycache__", ".git", ".venv", ".venv-mac", "venv", "env",
    ".pytest_cache", ".ipynb_checkpoints", ".playwright-mcp",
    "node_modules", ".DS_Store",
}

# Nhánh bỏ qua, tính từ thư mục gốc dự án.
SKIP_TREES = {
    "data/raw",      # 2,3 GB, tải lại được từ nguồn công bố
    "dist",          # gói nén để tải lên Kaggle, dựng lại được
    "references",    # bản PDF có bản quyền, tra lại được theo DOI
}

SKIP_SUFFIXES = {".zip", ".pyc", ".pyo"}

# Tệp bí mật: không bao giờ chép sang bản sao lưu. Tệp tokens.json dưới đây do
# một máy chủ MCP ghi nhầm bằng đường dẫn kiểu Windows nên nó rơi thẳng vào
# thư mục gốc dự án; nó chứa access token và refresh token còn hiệu lực.
SKIP_NAME_FRAGMENTS = ("tokens.json", "kaggle.json", "secrets.toml",
                       ".env", ".pem", ".key")


def should_skip(path: Path) -> bool:
    """Cho biết đường dẫn có thuộc nhóm loại khỏi bản bàn giao hay không."""
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if any(relative == tree or relative.startswith(tree + "/")
           for tree in SKIP_TREES):
        return True
    if any(part in SKIP_DIRECTORIES for part in path.parts):
        return True
    if any(fragment in path.name for fragment in SKIP_NAME_FRAGMENTS):
        return True
    return path.suffix in SKIP_SUFFIXES


def collect() -> list[Path]:
    """Thu thập các tệp được phép đưa vào bản sao lưu."""
    return sorted(p for p in PROJECT_ROOT.rglob("*")
                  if p.is_file() and not should_skip(p))


def digest(path: Path) -> str:
    """Tính mã SHA-256 để kiểm tra tệp sau khi sao lưu."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    """Tạo bản sao lưu hoàn thiện kèm bản kê và mã băm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liet-ke", action="store_true",
                        help="chỉ liệt kê và tính dung lượng, không chép")
    parser.add_argument("--dich", metavar="THU_MUC",
                        help="thư mục gốc chứa bản sao lưu, mặc định là "
                             f"{BACKUP_ROOT}. Dùng khi muốn sao lưu sang ổ khác")
    arguments = parser.parse_args()
    backup_root = Path(arguments.dich) if arguments.dich else BACKUP_ROOT

    files = collect()
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files):,} tệp, {total / 1e9:.3f} GB")

    if arguments.liet_ke:
        by_top: dict[str, int] = {}
        for path in files:
            top = path.relative_to(PROJECT_ROOT).parts[0]
            by_top[top] = by_top.get(top, 0) + path.stat().st_size
        for name, size in sorted(by_top.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<24} {size / 1e6:>10.1f} MB")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    destination = backup_root / stamp
    if destination.exists():
        print(f"Thư mục {destination} đã tồn tại, dừng để không ghi đè.")
        return 1
    destination.mkdir(parents=True)

    manifest = {}
    for path in files:
        relative = path.relative_to(PROJECT_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[relative.as_posix()] = {
            "sha256": digest(path), "bytes": path.stat().st_size}

    (destination / "BAN_KE_SAO_LUU.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(PROJECT_ROOT),
            "file_count": len(manifest),
            "total_bytes": total,
            "skipped_trees": sorted(SKIP_TREES),
            "files": manifest,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Đã sao lưu {len(manifest):,} tệp vào {destination}")
    print("Bản kê mã băm nằm ở BAN_KE_SAO_LUU.json trong cùng thư mục.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
