#!/usr/bin/env python3
"""
Dựng tài liệu kết quả đo lường từ các tệp JSON của những lần chạy thật.

Mọi con số trong tài liệu sinh ra đều đọc thẳng từ tệp kết quả của lần chạy
thật, không chép tay, nên không có nguy cơ sai lệch giữa báo cáo và dữ liệu
gốc. Chạy lại script sau mỗi notebook để tài liệu luôn khớp.

    python3 scripts/tong_hop_ket_qua.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tgn_nids.utils.figures import trivial_f1  # noqa: E402
from tgn_nids.utils.format import PERCENT_DECIMALS  # noqa: E402
from tgn_nids.utils.format import percent as format_percent  # noqa: E402

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "docs", "KET_QUA_DO_LUONG.md")

# Chỉ số nhị phân và chỉ số đa lớp phải đọc bằng hai bộ khoá khác nhau. Trộn
# lẫn chúng chính là lỗi đã làm hỏng bảng đối sánh của giai đoạn trước.
# f1_macro nằm trong danh sách binary vì GraphIDS và Anomal-E công bố chỉ số
# này cho bài toán hai lớp. Thiếu nó thì không đặt cạnh số liệu của họ được.
BINARY_METRICS = ["f1", "f1_macro", "precision", "recall", "far", "pr_auc",
                  "roc_auc", "accuracy", "balanced_accuracy"]
MULTICLASS_METRICS = ["f1_macro", "precision_macro", "recall_macro",
                      "far_macro", "accuracy"]


# Những sửa đổi làm kết quả cũ hết hiệu lực. Chỉ liệt kê sửa đổi thật sự đổi
# kết quả, chứ không phải mọi lần nâng số hiệu: đánh dấu tràn lan thì cảnh báo
# mất tác dụng, còn bỏ sót thì có nguy cơ trích dẫn số liệu đã hỏng.
#
# Mỗi mục gồm dòng đánh số mà nó thuộc về, phiên bản mang bản sửa, phạm vi ảnh
# hưởng, và lý do.
#
# Trường `line` là bắt buộc vì dự án có hai dòng đánh số phiên bản không liên
# tục với nhau. Bộ thí nghiệm bốn notebook chạy trên dòng 2.x; kiến trúc hai
# giai đoạn hiện hành là một bản viết lại và bắt đầu lại từ 1.0.0. So sánh
# thẳng hai dòng cho kết quả vô nghĩa: 1.1.0 nhỏ hơn 2.1.0 về mặt số học, nên
# mọi kết quả của kiến trúc hiện hành sẽ bị đánh dấu hết hiệu lực bởi những bản
# sửa mà chính nó đã mang sẵn từ đầu.
INVALIDATIONS = [
    {
        "line": 2,
        "since": (2, 7, 0),
        "scope": lambda r: r["config"]["mode"] == "supervised",
        "reason": "bộ phân loại trước đây không nhìn đặc trưng của chính cạnh "
                  "đang xét, nên không có thông tin nào để phân biệt loại tấn "
                  "công",
    },
    {
        "line": 2,
        "since": (2, 6, 0),
        "scope": lambda r: r["config"]["protocol"] in (
            "graphids", "random_stratified", "anomal_e"),
        "reason": "các tập bị xếp theo lớp tấn công thay vì theo trình tự "
                  "thời gian, khiến mỗi lô gần như thuần một lớp",
    },
    {
        "line": 2,
        "since": (2, 3, 0),
        "scope": lambda r: r["config"]["mode"] == "self_supervised",
        "reason": "bộ giải mã tự giám sát được viết lại sang cơ chế tái thiết "
                  "có che, bản cũ bị chặn trần ở PR-AUC 67,84%",
    },
    {
        "line": 2,
        "since": (2, 1, 0),
        "scope": lambda _result: True,
        "reason": "trọng số tốt nhất trước đây không được nạp lại trước khi "
                  "đánh giá, nên chỉ số thuộc về vòng lặp cuối",
    },
]


def as_tuple(version: str):
    """Chuyển chuỗi phiên bản thành bộ số nguyên để so sánh ổn định."""
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, ValueError):
        return (0,)


def invalidations_for(result):
    """Những sửa đổi khiến một kết quả cụ thể hết hiệu lực.

    Một quy tắc chỉ áp cho kết quả thuộc **cùng dòng đánh số** với nó. Kết quả
    của dòng khác không nằm trong tầm áp dụng, vì số hiệu của hai dòng không so
    sánh được với nhau.
    """
    version = as_tuple(result.get("source_version", "0"))
    return [rule for rule in INVALIDATIONS
            if version[0] == rule["line"] and version < rule["since"]
            and rule["scope"](result)]


def current_source_version() -> str:
    """Phiên bản mã nguồn hiện tại, dùng để phát hiện kết quả đã lỗi thời."""
    path = os.path.join(PROJECT_ROOT, "src", "tgn_nids", "__init__.py")
    with open(path, encoding="utf-8") as handle:
        match = re.search(r'SOURCE_VERSION\s*=\s*"([^"]+)"', handle.read())
    return match.group(1) if match else "?"


CURRENT_VERSION = current_source_version()


def percent(value, digits=PERCENT_DECIMALS):
    """Hiển thị một tỷ lệ dưới dạng phần trăm theo đúng quy tắc mục 2.4.

    Dấu thập phân là dấu phẩy theo quy ước tiếng Việt, giống hệt phần thân bài
    của báo cáo, để hai nơi không hiện cùng một con số theo hai kiểu khác nhau.

    Gọi thẳng vào `tgn_nids.utils.format` thay vì tự định dạng. Bản tự định
    dạng dùng hai chữ số và nhân trực tiếp với 100, nên một tỷ lệ cảnh báo sai
    0,004158% hiện ra là `0.00%`, tức một hệ thống có báo động giả trông như
    không hề có; còn diện tích dưới đường ROC 99,999189% hiện ra là `100.00%`,
    tức một bộ phân loại chưa hoàn hảo trông như hoàn hảo.
    """
    return format_percent(value, decimals=digits)


def so_nguyen(value):
    """Số đếm, dấu phân cách nghìn theo quy ước tiếng Việt là dấu chấm.

    Không dùng `f"{value:,}"` của Python, vì nó đặt dấu phẩy làm dấu phân cách
    nghìn theo quy ước tiếng Anh. Trong một tệp mà mọi phần trăm đều dùng dấu
    phẩy làm dấu thập phân, `236,549` đọc thành hai trăm ba mươi sáu phẩy năm
    bốn chín thay vì hai trăm ba mươi sáu nghìn năm trăm bốn mươi chín.
    """
    return "{:,}".format(int(value)).replace(",", ".")


def signed_percent(value):
    """Như percent nhưng luôn mang dấu, để phân biệt hơn mốc với kém mốc."""
    if not isinstance(value, (int, float)):
        return format_percent(value)
    dau = "+" if value >= 0 else "-"
    return dau + format_percent(abs(value))


def trivial_baseline(metrics):
    """Mốc tầm thường của tập kiểm thử, và khoảng cách từ F1 đo được tới mốc.

    Tỷ lệ mẫu dương lấy từ chính số mẫu của tập kiểm thử ghi trong tệp kết
    quả, không lấy từ tỷ lệ của toàn bộ dữ liệu: mốc phải tính trên đúng tập
    đã sinh ra con số F1 đặt cạnh nó.

    Trả về (None, None) khi tệp kết quả không ghi đủ số mẫu hai lớp. Không
    thay bằng số mặc định, vì một mốc bịa còn tệ hơn không có mốc.
    """
    nguon = metrics.get("binary_equivalent", metrics)
    duong = nguon.get("support_attack")
    am = nguon.get("support_benign")
    f1 = metrics.get("f1", nguon.get("f1"))
    if not isinstance(duong, (int, float)) or not isinstance(am, (int, float)):
        return None, None
    tong = duong + am
    if tong <= 0:
        return None, None
    moc = trivial_f1(duong / tong)
    return moc, (f1 - moc if isinstance(f1, (int, float)) else None)


def load_results():
    """Đọc mọi tệp result_*.json, gom theo thư mục chứa."""
    grouped = defaultdict(list)
    pattern = os.path.join(RESULTS_ROOT, "**", "result_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        # Thư mục lưu trữ chứa kết quả đã bị thay thế, giữ làm chứng cứ chứ
        # không đưa vào tài liệu số liệu. Trộn chúng vào đây là mời gọi trích
        # nhầm. Tên thư mục do collect_kaggle_results.ARCHIVE_PREFIX quy định
        # và bắt đầu bằng dấu gạch dưới, nên phải so đúng tiền tố đó.
        if os.sep + "_archive" in path:
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        relative = os.path.relpath(os.path.dirname(path), RESULTS_ROOT)
        grouped[relative.split(os.sep)[0]].append((path, payload))
    return grouped


def is_multiclass(result):
    """Nhận biết kết quả nhiều lớp qua bảng chỉ số theo lớp."""
    return "per_class" in result["metrics"]


def convergence_note(result):
    """Cảnh báo nếu lần chạy hết ngân sách vòng lặp mà chưa dừng sớm."""
    config = result["config"]
    ran = len(result["history"])
    budget, patience = config["num_epochs"], config["early_stopping_patience"]
    if ran >= budget and result["best_epoch"] > budget - patience:
        return f"**chưa hội tụ**, vòng tốt nhất {result['best_epoch']}/{budget}"
    return "đã hội tụ"


def write_run_section(out, name, entries):
    """Ghi một mục tổng hợp cho nhóm kết quả cùng thư mục chạy."""
    out.append(f"\n## Thư mục `{name}`\n")

    # Chỉ cảnh báo những kết quả thật sự bị một sửa đổi làm hết hiệu lực, chứ
    # không cảnh báo mọi kết quả có số hiệu phiên bản cũ hơn bản hiện tại.
    affected = defaultdict(list)
    for _, r in entries:
        for rule in invalidations_for(r):
            affected[rule["reason"]].append(r["config"]["name"])
    if affected:
        out.append("> **KHÔNG DÙNG ĐỂ ĐỐI SÁNH.** Những cấu hình sau chạy trên "
                   "bản mã nguồn có lỗi đã được sửa về sau, cần chạy lại:\n>")
        for reason, names in affected.items():
            listing = ", ".join(f"`{n}`" for n in sorted(names))
            out.append(f"> - {listing} — {reason}.")
        out.append(">")

    intact = [r["config"]["name"] for _, r in entries if not invalidations_for(r)]
    stale = [r for _, r in entries if r.get("source_version") != CURRENT_VERSION
             and not invalidations_for(r)]
    if stale:
        mo_dau = "Các cấu hình còn lại" if affected else "Toàn bộ cấu hình ở đây"
        out.append(
            f"> {mo_dau} chạy trên bản cũ hơn {CURRENT_VERSION} nhưng **vẫn "
            f"còn hiệu lực**, vì không sửa đổi nào ảnh hưởng tới chúng.\n")
    elif affected and intact:
        out.append("")

    out.append("### Điều kiện chạy\n")
    out.append("| Cấu hình | Mã nguồn | Protocol | Task | Mode | "
               "Lấy mẫu | Train | Test | Edge features |")
    out.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for _, r in entries:
        c = r["config"]
        out.append(
            f"| `{c['name']}` | {r.get('source_version', '—')} | {c['protocol']} "
            f"| {c['task']} | {c['mode']} | {c.get('sample_ratio', 1.0):.0%} "
            f"| {so_nguyen(r['split_stats']['train']['count'])} "
            f"| {so_nguyen(r['split_stats']['test']['count'])} | {r['edge_dim']} |")

    out.append("\n### Huấn luyện\n")
    out.append("| Cấu hình | Vòng đã chạy | Ngân sách | Vòng tốt nhất | "
               "Val loss | Trạng thái | Thời gian |")
    out.append("|---|---:|---:|---:|---:|---|---:|")
    for _, r in entries:
        c = r["config"]
        loss = r.get("best_val_loss")
        out.append(
            f"| `{c['name']}` | {len(r['history'])} | {c['num_epochs']} "
            f"| {r['best_epoch']} | {loss:.5f} | {convergence_note(r)} "
            f"| {r['training_time_seconds']:.0f}s |")

    binary = [(p, r) for p, r in entries if not is_multiclass(r)]
    if binary:
        out.append("\n### Chỉ số, bài toán binary\n")
        out.append(
            "Cột **mốc tầm thường** là điểm F1 mà một bộ phân loại gán nhãn "
            "tấn công cho mọi mẫu sẽ đạt được, tính bằng `2·rate/(1+rate)` "
            "với `rate` là tỷ lệ mẫu tấn công của tập kiểm thử. Mục 2.5 bộ "
            "quy tắc dự án buộc mọi chỉ số F1 nhị phân phải đứng cạnh mốc "
            "này: thiếu nó, một giá trị F1 cao vẫn có thể nằm dưới mức đoán "
            "bừa không cần mô hình.\n")
        out.append("| Cấu hình | " + " | ".join(BINARY_METRICS)
                   + " | mốc tầm thường | f1 trừ mốc |")
        out.append("|---" * (len(BINARY_METRICS) + 3) + "|")
        for _, r in binary:
            values = " | ".join(percent(r["metrics"].get(m))
                                for m in BINARY_METRICS)
            moc, chenh = trivial_baseline(r["metrics"])
            out.append(f"| `{r['config']['name']}` | {values} "
                       f"| {percent(moc)} | {signed_percent(chenh)} |")

    multi = [(p, r) for p, r in entries if is_multiclass(r)]
    if multi:
        out.append("\n### Chỉ số, bài toán multiclass\n")
        # Số lớp khác nhau giữa các cấu hình, nên câu mô tả phải đọc từ chính
        # dữ liệu thay vì ghi cứng một con số. Ghi cứng "mười lớp" là sai ngay
        # khi bảng có thêm một cấu hình mười lăm lớp, và cái sai đó không lộ ra
        # vì bảng vẫn hiện đủ mọi hàng.
        class_counts = sorted({len(r.get("class_names") or []) for _, r in multi})
        counts_text = " hoặc ".join(str(c) for c in class_counts if c)
        out.append(
            f"Giá trị trung bình macro trên toàn bộ các lớp của từng bộ dữ "
            f"liệu. Số lớp không giống nhau giữa các cấu hình trong bảng, hiện "
            f"là {counts_text}, nên hai giá trị macro của hai bộ khác nhau "
            f"không phải hai phép đo cùng thang.\n")
        out.append(
            "Cột cuối là **recall của lớp tấn công**, đo trên bài toán nhị "
            "phân quy đổi: trong toàn bộ luồng tấn công thật, mô hình gọi "
            "đúng tên một lớp tấn công nào đó cho bao nhiêu phần. Con số này "
            "bắt buộc đi kèm mọi chỗ hiển thị Macro "
            "F1, vì Macro F1 có thể cao nhờ riêng lớp lành tính trong khi bộ "
            "phát hiện đang bỏ sót phần lớn số cuộc tấn công.\n")
        out.append("| Cấu hình | " + " | ".join(MULTICLASS_METRICS)
                   + " | recall lớp tấn công (nhị phân quy đổi) |")
        out.append("|---" * (len(MULTICLASS_METRICS) + 2) + "|")
        for _, r in multi:
            values = " | ".join(percent(r["metrics"].get(m))
                                for m in MULTICLASS_METRICS)
            # Lấy từ nhánh binary_equivalent chứ không từ recall_macro: hai
            # đại lượng khác nhau, và mục 2.7 bộ quy tắc cấm gán chỉ số đo
            # trên bài toán này cho bài toán khác.
            recall_tan_cong = r["metrics"].get("binary_equivalent", {}).get("recall")
            out.append(f"| `{r['config']['name']}` | {values} "
                       f"| {percent(recall_tan_cong)} |")

        for _, r in multi:
            out.append(f"\n**Chi tiết từng lớp, `{r['config']['name']}`**\n")
            out.append("| Lớp | Precision | Recall | F1 | FAR | Support |")
            out.append("|---|---:|---:|---:|---:|---:|")
            # Cùng đơn vị phần trăm với mọi bảng khác trong tài liệu. Bản
            # trộn hai đơn vị buộc người đọc tự nhớ rằng 0,973 ở bảng này và
            # 97,300% ở bảng trên là cùng một đại lượng.
            for entry in r["metrics"]["per_class"]:
                out.append(
                    f"| {entry['name']} | {percent(entry['precision'])} "
                    f"| {percent(entry['recall'])} | {percent(entry['f1'])} "
                    f"| {percent(entry['far'])} | {so_nguyen(entry['support'])} |")

    with_points = [(p, r) for p, r in entries if r.get("operating_points")]
    if with_points:
        out.append("\n### Điểm làm việc\n")
        out.append("Ngưỡng luôn chọn trên tập validation rồi mới áp lên tập "
                   "test. Điểm `far_0.0009` ứng với mức FAR 0,09% mà "
                   "TE-G-SAGE công bố.\n")
        out.append("| Cấu hình | Điểm | Threshold | Precision | Recall "
                   "| F1 | FAR | Đạt ràng buộc |")
        out.append("|---|---|---:|---:|---:|---:|---:|---|")
        for _, r in with_points:
            for label, point in r["operating_points"].items():
                threshold = ("—" if point["threshold"] is None
                             else f"{point['threshold']:.5f}")
                met = {True: "có", False: "**không**", None: "—"}[
                    point.get("dat_rang_buoc")]
                out.append(
                    f"| `{r['config']['name']}` | {label} | {threshold} "
                    f"| {percent(point['precision'])} | {percent(point['recall'])} "
                    f"| {percent(point['f1'])} | {percent(point['far'], 4)} "
                    f"| {met} |")

    weights = [(p, r) for p, r in entries if r.get("class_weights")]
    if weights:
        out.append("\n### Trọng số lớp đã dùng\n")
        for _, r in weights:
            pairs = ", ".join(
                f"{n}={w}" for n, w in zip(
                    r["class_names"] if is_multiclass(r) else ["Benign", "Attack"],
                    r["class_weights"]))
            out.append(f"- `{r['config']['name']}`: {pairs}")


def write_seed_summary(out, entries):
    """Gom các cấu hình chỉ khác nhau hạt giống ngẫu nhiên."""
    groups = defaultdict(list)
    for _, r in entries:
        name = r["config"]["name"]
        if "_s" not in name:
            continue
        head, tail = name.split("_s", 1)
        groups[head + "_" + tail.split("_", 1)[1]].append(r)
    if not any(len(v) > 1 for v in groups.values()):
        return

    import statistics
    out.append("\n### Gom theo hạt giống ngẫu nhiên\n")
    out.append("Cột `n` là số hạt giống thực tế. Độ lệch chuẩn chỉ có nghĩa "
               "khi `n` từ 2 trở lên.\n")
    out.append("| Phương án | n | PR-AUC | ROC-AUC | F1 | FAR |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name in sorted(groups):
        runs = groups[name]
        cells = []
        for metric in ("pr_auc", "roc_auc", "f1", "far"):
            values = [r["metrics"][metric] for r in runs
                      if isinstance(r["metrics"].get(metric), float)]
            if not values:
                cells.append("—")
            elif len(values) == 1:
                cells.append(percent(values[0]))
            else:
                cells.append(f"{statistics.mean(values) * 100:.2f} ± "
                             f"{statistics.pstdev(values) * 100:.2f}")
        out.append(f"| `{name}` | {len(runs)} | " + " | ".join(cells) + " |")


def build_document():
    """Dựng toàn bộ nội dung Markdown từ các kết quả đã lưu."""
    grouped = load_results()
    out = [
        "# Kết quả đo lường",
        "",
        "**Tài liệu này được sinh tự động** bởi `scripts/tong_hop_ket_qua.py`,",
        "đọc thẳng từ các tệp `result_*.json` của lần chạy thật. Không chép tay con",
        "số nào, nên tài liệu luôn khớp dữ liệu gốc. Chạy lại script sau mỗi",
        "notebook để cập nhật.",
        "",
        "Phần diễn giải và các phép đo không sinh ra từ notebook nằm ở cuối, trong",
        "mục *Các phép đo bổ trợ*. Quy tắc chọn ngưỡng và bảng tham số đã chốt "
        "xem `docs/SO_LIEU_DAU_RA_CHI_TIET.md`.",
        "",
        f"Số thí nghiệm đã có kết quả: **{sum(len(v) for v in grouped.values())}**.",
    ]
    for name in sorted(grouped):
        entries = sorted(grouped[name], key=lambda item: item[1]["config"]["name"])
        write_run_section(out, name, entries)
        write_seed_summary(out, entries)
    return "\n".join(out) + "\n"


MANUAL_SECTION = """

---

# Các phép đo bổ trợ

Những con số dưới đây không sinh ra từ notebook mà đo riêng trong quá trình
chẩn đoán. Mỗi mục ghi rõ cách đo để kiểm chứng lại được.

## 1. Bài toán binary trên NF-UNSW-NB15-v3 đã được giải sẵn

Đo trên toàn bộ 2.365.424 luồng, không huấn luyện gì:

| Phép đo | Kết quả |
|---|---|
| Địa chỉ nguồn chỉ sinh lưu lượng tấn công | 4 địa chỉ, dải `175.45.176.0/30` |
| Địa chỉ nguồn chỉ sinh lưu lượng benign | 36 |
| Địa chỉ nguồn sinh cả hai loại | **0** |
| Quy tắc chỉ xét địa chỉ nguồn | **sai 0 luồng trên 2.365.424** |
| `MAX_TTL` dùng một mình làm bộ phân loại | AUC **0,9986** |
| `MIN_TTL` dùng một mình | AUC 0,9985 |

Phân bố TTL: lưu lượng benign tập trung ở giá trị 32, lưu lượng tấn công tập
trung ở 255.

## 2. Mốc tham chiếu bảng, không dùng đồ thị

Toàn bộ dữ liệu, đúng giao thức chia tập của TE-G-SAGE, chỉ dùng 49 thuộc tính
luồng mạng, không biết gì về địa chỉ mạng:

| Mô hình | F1 | FAR | PR-AUC |
|---|---:|---:|---:|
| Cây quyết định sâu 3, đủ thuộc tính | **99,79%** | 0,0333% | 99,80% |
| Cây quyết định sâu 3, bỏ `MIN_TTL` và `MAX_TTL` | **99,55%** | 0,0661% | 99,14% |
| Hồi quy logistic, đủ thuộc tính | 99,49% | 0,0305% | 99,80% |
| *TE-G-SAGE công bố* | *99,52%* | *0,0900%* | *không công bố* |
| *GraphIDS công bố* | *không công bố* | *không công bố* | *99,98%* |

Toàn bộ quy tắc của một cây sâu hai tầng:

```
MIN_TTL <= 1.52  ->  MAX_IP_PKT_LEN <= 1.27 ? Benign : Attack
MIN_TTL >  1.52  ->  PROTOCOL       <= -3.14 ? Benign : Attack
```

Phép đo này nằm trong notebook 05 nên tái lập được.

## 3. Ablation kiến trúc

Bốn cấu hình đã chạy trên NF-UNSW-NB15-v3, mỗi cấu hình huấn luyện lại từ đầu.
Kết quả nằm ở `models/saved/twoDTS_ablation/`, bảng đầy đủ kèm điều kiện đo ở
`docs/SO_LIEU_DAU_RA_CHI_TIET.md`, mục thực nghiệm cắt bỏ kiến trúc.

Mức đóng góp đo được, tính bằng điểm phần trăm Macro F1 so với mốc đối chiếu:
danh tính đỉnh 2,827 điểm, phép tổng hợp lân cận 0,492 điểm, tám cột đặc trưng
thời gian IAT 0,320 điểm.

Cả ba giá trị đo trên **một hạt giống duy nhất**, nên hai con số nhỏ chưa tách
khỏi dao động giữa các lần chạy và phải trích kèm giới hạn phát hiện. Mở rộng
sang nhiều hạt giống thuộc Hướng phát triển, ghi ở mục 6
`docs/THUC_NGHIEM_CAT_BO_KIEN_TRUC.md`.

Phép triệt tiêu từng đặc trưng ở mục trên là công cụ giải thích, không phải
ablation kiến trúc, và không trả lời được câu hỏi về đóng góp của thành phần
kiến trúc.

## 4. Số liệu do các nghiên cứu đối sánh công bố

Trích trực tiếp từ bản gốc trong `references/`.

| Nghiên cứu | Bộ | Bài toán | Chỉ số |
|---|---|---|---|
| GraphIDS | v3 | binary | Macro F1 99,61% · PR-AUC 99,98% |
| GraphIDS | v2 | binary | Macro F1 92,64% · PR-AUC 81,16% |
| TE-G-SAGE | v3 | binary | P 99,06% · R 99,99% · F1 99,52% · FAR 0,09% |
| TE-G-SAGE | v3 | multiclass | Acc 95,59% · P 49,42% · R 62,74% · Macro F1 49,06% · FAR 0,45% |
| Anomal-E + IF | v2 | binary | Acc 98,66% · Macro F1 92,35% · DR 98,77% |

TE-G-SAGE theo lớp, bài toán multiclass: Analysis 0,305 · Backdoor 0,071 ·
Benign 1,000 · DoS 0,260 · Exploits 0,613 · Fuzzers 0,501 · Generic 0,796 ·
Reconnaissance 0,579 · Shellcode 0,221 · Worms 0,500.

TCG-IDS chưa có bản gốc nên **không trích số nào**.
"""


if __name__ == "__main__":
    document = build_document() + MANUAL_SECTION
    # newline="\n" để tệp sinh ra giống nhau trên cả Windows lẫn macOS. Chế độ
    # văn bản mặc định của Windows đổi mọi dấu xuống dòng thành CRLF, nên chạy
    # script trên hai hệ điều hành sẽ cho hai tệp khác nhau ở từng dòng dù nội
    # dung không đổi. `.gitattributes` của dự án cũng chốt `*.md text eol=lf`.
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    print(f"Đã ghi: {OUTPUT_PATH}")
    print(f"Dung lượng: {len(document) / 1024:.1f} KB, "
          f"{document.count(chr(10))} dòng")
