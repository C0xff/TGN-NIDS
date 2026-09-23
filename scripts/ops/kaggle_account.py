"""Chuyển qua lại giữa nhiều tài khoản Kaggle.

Kaggle CLI chỉ đọc đúng một tệp `~/.kaggle/kaggle.json`, nên muốn dùng nhiều
tài khoản thì phải thay tệp đó. Script này giữ mỗi tài khoản trong một tệp
riêng đặt tên theo chính tài khoản, rồi chép tệp được chọn vào vị trí mà CLI
đọc:

    ~/.kaggle/kaggle.<tên tài khoản>.json    bản lưu của từng tài khoản
    ~/.kaggle/kaggle.json                    bản đang hoạt động

Đặt tên theo tài khoản thay vì `kaggle_bak.json` để không bao giờ phải đoán tệp
nào là của ai — đây là chỗ dễ nhầm nhất khi hạn mức GPU của hai tài khoản khác
nhau và một lần chạy nhầm tốn vài giờ.

    python scripts/kaggle_account.py                  # xem danh sách
    python scripts/kaggle_account.py use julonao      # chuyển tài khoản
"""

import argparse
import json
import shutil
import stat
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
ACTIVE_FILE = KAGGLE_DIR / "kaggle.json"
STORE_PREFIX = "kaggle."
STORE_SUFFIX = ".json"


def read_username(path: Path):
    """Đọc tên tài khoản trong một tệp thông tin xác thực."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("username")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def stored_accounts() -> dict:
    """Liệt kê các tài khoản đã lưu, ánh xạ tên tài khoản sang đường dẫn tệp."""
    accounts = {}
    for path in sorted(KAGGLE_DIR.glob(f"{STORE_PREFIX}*{STORE_SUFFIX}")):
        if path.name == ACTIVE_FILE.name:
            continue
        username = read_username(path)
        if username:
            accounts[username] = path
    return accounts


def adopt_loose_files() -> list:
    """Đưa các tệp thông tin xác thực rời rạc về đúng quy ước đặt tên.

    Nhận cả `kaggle.json` đang hoạt động lẫn những tệp đặt tên tuỳ tiện kiểu
    `kaggle_bak.json`, đọc tên tài khoản bên trong rồi lưu lại thành
    `kaggle.<tên tài khoản>.json`.
    """
    adopted = []
    candidates = [path for path in KAGGLE_DIR.glob("*.json")
                  if not (path.name.startswith(STORE_PREFIX)
                          and path.name.count(".") == 2)]
    for path in candidates:
        username = read_username(path)
        if not username:
            continue
        target = KAGGLE_DIR / f"{STORE_PREFIX}{username}{STORE_SUFFIX}"
        if target.exists():
            continue
        shutil.copy2(path, target)
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)
        adopted.append((path.name, target.name))
    return adopted


def activate(username: str) -> None:
    """Đặt một tài khoản làm tài khoản đang hoạt động."""
    accounts = stored_accounts()
    if username not in accounts:
        raise SystemExit(f"Chưa lưu tài khoản '{username}'. "
                         f"Đã có: {', '.join(accounts) or 'chưa có tài khoản nào'}")
    # Tệp access_token là phiên đăng nhập OAuth, và với một số tài khoản thì
    # CHÍNH NÓ mới là thứ xác thực chứ không phải khoá trong kaggle.json —
    # khoá trong tệp json có thể đã bị thu hồi mà không có dấu hiệu nào. Vì
    # vậy phải cất giữ access_token theo từng tài khoản chứ không xoá chung:
    # xoá đi thì tài khoản phụ thuộc vào nó sẽ mất quyền truy cập.
    token_file = KAGGLE_DIR / "access_token"
    previous = read_username(ACTIVE_FILE) if ACTIVE_FILE.exists() else None

    if token_file.exists() and previous and previous != username:
        stash = KAGGLE_DIR / f"access_token.{previous}"
        shutil.move(str(token_file), str(stash))
        print(f"Đã cất phiên đăng nhập của '{previous}' thành {stash.name}")

    shutil.copy2(accounts[username], ACTIVE_FILE)
    ACTIVE_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

    restored = KAGGLE_DIR / f"access_token.{username}"
    if restored.exists():
        shutil.move(str(restored), str(token_file))
        print(f"Đã khôi phục phiên đăng nhập của '{username}'")

    print(f"Tài khoản đang hoạt động: {username}")


def show() -> None:
    """In danh sách tài khoản đã lưu và tài khoản đang hoạt động."""
    active = read_username(ACTIVE_FILE) if ACTIVE_FILE.exists() else None
    accounts = stored_accounts()
    if not accounts:
        print("Chưa lưu tài khoản nào trong ~/.kaggle/")
        return
    print(f"{'Tài khoản':<24s} {'Trạng thái':<16s} Tệp lưu")
    for username, path in accounts.items():
        state = "đang hoạt động" if username == active else ""
        print(f"{username:<24s} {state:<16s} {path.name}")
    if active and active not in accounts:
        print(f"\nLưu ý: kaggle.json đang là tài khoản '{active}' nhưng chưa "
              f"được lưu thành tệp riêng.")


def main() -> None:
    """Thực hiện lệnh quản lý tài khoản và tài nguyên Kaggle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="list",
                        choices=["list", "use"])
    parser.add_argument("username", nargs="?")
    arguments = parser.parse_args()

    if not KAGGLE_DIR.exists():
        raise SystemExit(f"Không tìm thấy thư mục {KAGGLE_DIR}")

    adopted = adopt_loose_files()
    for source, target in adopted:
        print(f"Đã lưu {source} thành {target}")
    if adopted:
        print()

    if arguments.command == "use":
        if not arguments.username:
            raise SystemExit("Thiếu tên tài khoản. Ví dụ: use julonao")
        activate(arguments.username)
    else:
        show()


if __name__ == "__main__":
    main()
