"""Tải kết quả ba notebook thí nghiệm từ Kaggle và sắp vào models/saved.

Cấu trúc thư mục sau khi thu gom giữ **nguyên si** cây thư mục mà notebook sinh
ra trên Kaggle, cho mỗi nhóm thí nghiệm:

    models/saved/<nhóm>/
        <ô mã>/models/model_<tên>.pt          trọng số kèm tham số suy luận
        <ô mã>/models/predictions_<tên>.npz   nhãn thật, nhãn dự đoán, anomaly score
        <ô mã>/models/result_<tên>.json       toàn bộ chỉ số và cấu hình
        <ô mã>/figures/*.png|*.pdf            hình đánh giá
        <ô mã>/tables/*.csv                   bảng số liệu
        <ô mã>/metrics/*.json

Giữ nguyên cây là **bắt buộc**, không phải lựa chọn thẩm mỹ. Bảng MODEL_CONFIGS
của dashboard trỏ thẳng vào models/saved/<nhóm>/<ô mã>/models/<tệp>, còn
NB3_phan_tich.ipynb quét theo mẫu "*/models/result_*.json". Cả hai đều đi qua
cấp thư mục <ô mã>, nên một bố cục làm phẳng sẽ khiến chúng đọc ra rỗng **mà
không báo lỗi nào**: cả hai chỉ tìm không thấy tệp rồi trả danh sách trống.

Cấp thư mục <ô mã> còn phục vụ việc viết báo cáo: khi bóc tách kết quả thì biết
ngay mỗi hình, mỗi bảng sinh ra từ bước nào của notebook.

Mỗi lần thu gom sẽ **xoá sạch thư mục nhóm trước khi chép**, để kết quả của hai
lần chạy khác nhau không bao giờ nằm lẫn trong cùng một thư mục. Riêng thư mục
`_archive/` được giữ nguyên: đó là nơi lưu bản ghi của những lần chạy trước.

    python scripts/collect_kaggle_results.py                      # thu gom cả ba
    python scripts/collect_kaggle_results.py nb3                  # chỉ một nhóm
    python scripts/collect_kaggle_results.py nb1 --owner=yuto0xc  # tài khoản khác
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_ROOT = PROJECT_ROOT / "models" / "saved"

# Tài khoản mặc định. Dự án có lúc chạy trên hai tài khoản Kaggle khác nhau vì
# hạn mức GPU theo tuần, nên tên chủ sở hữu tách riêng thành tham số thay vì
# ghi cứng vào từng slug.
DEFAULT_OWNER = "julonao"

# Ánh xạ từ kernel trên Kaggle sang tên nhóm dùng trong models/saved.
# Tên nhóm phải trùng với NOTEBOOK_ID khai báo trong ô "Bố cục thư mục kết quả"
# của chính notebook đó, vì dashboard trỏ thẳng vào
# models/saved/<NOTEBOOK_ID>/<ô mã>/models. Lệch tên thì dashboard không nạp
# được mô hình nào, và nó không báo lỗi mà chỉ hiện danh sách rỗng.
KERNELS = {
    "nb1": {"kernel": "b-twodts-train-v2",
            "group": "twoDTS_train_v2"},
    "nb2": {"kernel": "b-twodts-train-v3",
            "group": "twoDTS_train_v3"},
    "nb3": {"kernel": "b-twodts-phan-tich",
            "group": "twoDTS_phan_tich"},
}

# Đuôi tệp không kéo về đĩa. Ba tệp .pth của mỗi lần chạy là cơ chế early stopping:
# runner ghi chúng mỗi khi gặp epoch tốt nhất rồi nạp lại ở cuối để khôi phục
# trọng số tốt nhất. Sau khi lần chạy kết thúc thì tệp .pt đã chứa đủ cả ba
# state_dict cộng preprocessor, node_mapping và danh mục nhãn, nên .pth không
# còn mang thêm thông tin học được nào. Chúng vẫn được ghi bình thường trong
# thư mục kết quả của phiên Kaggle; chỉ không kéo về kho để kho gọn.
# Chốt ngày 03/09/2026. Lần chạy thứ ba bỏ lại 141 MB nhờ quy tắc này.
SKIP_SUFFIXES = (".pth",)

# Thư mục 99_export chỉ chứa bản sao của những tệp đã có ở nơi khác, cùng tệp
# nén tổng. Đã đối chiếu mã băm nội dung ngày 04/09/2026 trên cả hai lần chạy:
# trùng khớp hoàn toàn với outputs/, không tệp nào chỉ có ở một bên.
SKIP_DIR_NAMES = ("99_export",)

# Thư mục này chứa bản lưu của các lần chạy trước, không bị xoá khi thu gom kết
# quả mới. Tên bắt đầu bằng dấu gạch dưới để luôn nằm tách khỏi các thư mục kết
# quả của lần chạy hiện hành khi liệt kê.
ARCHIVE_PREFIX = "_archive"


def run_kaggle(*arguments) -> str:
    """Gọi Kaggle CLI, trả về đầu ra dạng văn bản.

    Mã thoát khác không được nêu ra ngay thay vì nuốt đi. Kaggle CLI trả lỗi
    xác thực và lỗi hết hạn mức bằng mã thoát chứ không bằng một dòng đầu ra
    nhận dạng được, nên nếu chỉ ghép stdout với stderr rồi trả về thì
    kernel_status đọc không ra trạng thái và báo UNKNOWN, còn download_output
    tạo một thư mục rỗng và kết thúc như thể đã tải xong.
    """
    result = subprocess.run(["kaggle", *arguments],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Lệnh 'kaggle {' '.join(arguments)}' thoát với mã "
            f"{result.returncode}.\n{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def kernel_status(slug: str) -> str:
    """Đọc trạng thái hiện tại của một kernel."""
    output = run_kaggle("kernels", "status", slug)
    for token in output.split():
        if token.startswith('"KernelWorkerStatus'):
            return token.strip('"').split(".")[-1]
    return "UNKNOWN"


def download_output(slug: str, destination: Path) -> None:
    """Tải toàn bộ output của kernel về một thư mục."""
    destination.mkdir(parents=True, exist_ok=True)
    run_kaggle("kernels", "output", slug, "-p", str(destination))


def extract_results_archive(download_dir: Path, extract_dir: Path) -> bool:
    """Giải nén tệp results_*.zip nếu có; trả về True khi giải nén được."""
    archives = list(download_dir.glob("results_*.zip"))
    if not archives:
        return False
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(extract_dir)
    return True


def collect_group(source_root: Path, group: str) -> dict:
    """Sắp toàn bộ artefact của một nhóm vào models/saved/<nhóm>.

    source_root là thư mục đã giải nén, chứa cây thư mục theo cell mà notebook
    sinh ra.
    """
    target = SAVED_ROOT / group

    # Xoá sạch kết quả cũ trước khi chép bản mới. Chép đè mà không xoá sẽ để
    # lại tệp của lần chạy trước: một lần chạy đổi tên cell hoặc bỏ bớt một cấu
    # hình sẽ khiến thư mục chứa lẫn lộn hai lần chạy, và không còn cách nào
    # phân biệt tệp nào thuộc lần nào khi trích số liệu vào báo cáo.
    if target.exists():
        removed = 0
        for entry in target.iterdir():
            # Thư mục lưu trữ có tiền tố before_change được giữ lại: đó là bản
            # ghi của những lần chạy trước, dùng để đối chiếu và chứng thực số
            # liệu đã báo cáo. Xoá chúng đi thì không còn cách nào kiểm lại một
            # con số đã trích vào báo cáo.
            if entry.is_dir() and entry.name == ARCHIVE_PREFIX:
                continue
            if entry.is_dir():
                removed += sum(1 for path in entry.rglob("*") if path.is_file())
                shutil.rmtree(entry)
            else:
                removed += 1
                entry.unlink()
        if removed:
            print(f"   đã xoá {removed} tệp kết quả cũ trong models/saved/{group}")
        archive = target / ARCHIVE_PREFIX
        if archive.exists():
            saved_runs = sorted(e.name for e in archive.iterdir() if e.is_dir())
            print(f"   giữ nguyên {ARCHIVE_PREFIX}/ với {len(saved_runs)} lần chạy cũ: "
                  f"{', '.join(saved_runs)}")
    target.mkdir(parents=True, exist_ok=True)

    counters = {"models": 0, "figures": 0, "tables": 0, "metrics": 0, "khác": 0}
    skipped = 0

    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(source_root)
        parts = relative.parts

        if any(part in SKIP_DIR_NAMES for part in parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            skipped += 1
            continue

        # Chép giữ nguyên đường dẫn tương đối. Đây là điểm khác cốt lõi so với
        # bản trước: không xếp lại tệp theo loại, vì cấp thư mục <ô mã> nằm
        # trong hợp đồng đường dẫn của dashboard và của NB3.
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

        # Đếm theo thư mục con để bản tóm tắt in ra vẫn đọc được như trước.
        for kind in ("models", "figures", "tables", "metrics"):
            if kind in parts:
                counters[kind] += 1
                break
        else:
            counters["khác"] += 1

    if skipped:
        print(f"   bỏ lại {skipped} tệp .pth theo quy tắc chốt 03/09/2026")

    # Kiểm ngay hợp đồng đường dẫn thay vì để dashboard và NB3 phát hiện muộn
    # bằng một danh sách rỗng không kèm lý do.
    found = list(target.glob("*/models/result_*.json"))
    if not found:
        raise SystemExit(
            f"Thu gom xong nhưng không có tệp nào khớp mẫu "
            f"models/saved/{group}/*/models/result_*.json. Dashboard và NB3 đều "
            f"quét theo mẫu này, nên chúng sẽ đọc ra rỗng. Kiểm lại bố cục thư "
            f"mục mà notebook sinh ra trước khi dùng kết quả này.")
    print(f"   hợp đồng đường dẫn: {len(found)} tệp result_*.json đúng mẫu "
          f"<ô mã>/models/")

    return counters


def summarise_group(group: str) -> None:
    """In bảng chỉ số của một nhóm, đọc từ các tệp result_*.json."""
    target = SAVED_ROOT / group
    rows = []
    for path in sorted(target.rglob("result_*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            # Báo ra chứ không bỏ qua im lặng. Một tệp kết quả hỏng mà biến mất
            # khỏi bảng tổng hợp trông y hệt một lần chạy chưa từng tồn tại, và
            # đó là kiểu thiếu sót không ai phát hiện khi đọc bảng.
            print(f"  CẢNH BÁO: không đọc được {path.name} ({error}); "
                  f"lần chạy này thiếu khỏi bảng tổng hợp")
            continue
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        history = data.get("history", []) or []
        best_epoch = data.get("best_epoch")
        rows.append({
            "name": config.get("name", path.stem[7:]),
            "task": config.get("task", "?"),
            "f1_macro": metrics.get("f1_macro"),
            "accuracy": metrics.get("accuracy"),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "budget": config.get("num_epochs"),
        })

    if not rows:
        print("   chưa có tệp result_*.json nào")
        return

    print(f"   {'Lần chạy':<36s} {'Task':<11s} {'Macro F1':>10s} {'Accuracy':>10s} "
          f"{'best/ran/ngân sách':>20s}")
    for row in rows:
        f1 = f"{row['f1_macro'] * 100:.4f}%" if row["f1_macro"] is not None else "-"
        accuracy = f"{row['accuracy'] * 100:.4f}%" if row["accuracy"] is not None else "-"
        progress = f"{row['best_epoch']}/{row['epochs_run']}/{row['budget']}"
        # Vòng tốt nhất nằm sát trần ngân sách nghĩa là mô hình còn đang tiến
        # bộ khi bị cắt, và chỉ số thu được là của mô hình chưa hội tụ đủ.
        warning = ""
        if row["best_epoch"] and row["budget"] and row["epochs_run"] >= row["budget"]:
            warning = "  <-- chạm trần ngân sách"
        print(f"   {row['name'][:36]:<36s} {row['task']:<11s} {f1:>10s} "
              f"{accuracy:>10s} {progress:>20s}{warning}")


def main() -> None:
    """Thu thập kết quả Kaggle về đúng cấu trúc thư mục của dự án."""
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    owner = DEFAULT_OWNER
    for flag in sys.argv[1:]:
        if flag.startswith("--owner="):
            owner = flag.split("=", 1)[1]

    requested = arguments or list(KERNELS)
    unknown = [key for key in requested if key not in KERNELS]
    if unknown:
        raise SystemExit(f"Không nhận ra nhóm: {', '.join(unknown)}. "
                         f"Chọn trong: {', '.join(KERNELS)}")

    SAVED_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Tài khoản Kaggle: {owner}")

    for key in requested:
        entry = KERNELS[key]
        slug, group = f"{owner}/{entry['kernel']}", entry["group"]
        status = kernel_status(slug)
        print(f"\n{key.upper()}  {slug}  [{status}]")

        if status != "COMPLETE":
            print("   bỏ qua, kernel chưa chạy xong")
            continue

        with tempfile.TemporaryDirectory() as workspace:
            download_dir = Path(workspace) / "download"
            extract_dir = Path(workspace) / "extract"

            print("   đang tải output từ Kaggle")
            download_output(slug, download_dir)

            if extract_results_archive(download_dir, extract_dir):
                source_root = extract_dir
            else:
                # Không có tệp nén thì lấy thẳng cây thư mục outputs đã tải.
                candidates = list(download_dir.glob("outputs/*"))
                if not candidates:
                    print("   không tìm thấy kết quả nào trong output")
                    continue
                source_root = candidates[0]

            counters = collect_group(source_root, group)

        print(f"   đã sắp vào models/saved/{group}: "
              f"{counters['models']} tệp mô hình, {counters['figures']} hình, "
              f"{counters['tables']} bảng, {counters['metrics']} bản kê khai")
        summarise_group(group)


if __name__ == "__main__":
    main()
