#!/usr/bin/env python3
"""
Dựng tài liệu kết quả đo lường từ các tệp JSON của những lần chạy thật.

Mọi con số đọc thẳng từ tệp kết quả, không chép tay. Chạy lại sau mỗi notebook.

    python3 scripts/analysis/tong_hop_ket_qua.py
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

# Chỉ số nhị phân và đa lớp đọc bằng hai bộ khoá riêng. f1_macro có trong bộ
# nhị phân vì GraphIDS và Anomal-E công bố chỉ số này cho bài toán hai lớp.
BINARY_METRICS = ["f1", "f1_macro", "precision", "recall", "far", "pr_auc",
                  "roc_auc", "accuracy", "balanced_accuracy"]
MULTICLASS_METRICS = ["f1_macro", "precision_macro", "recall_macro",
                      "far_macro", "accuracy"]


# Những sửa đổi làm kết quả cũ hết hiệu lực, chỉ gồm sửa đổi thật sự đổi kết quả.
# Trường `line` tách hai dòng đánh số phiên bản: bộ bốn notebook cũ dùng 2.x, còn
# kiến trúc hai giai đoạn viết lại từ 1.0.0, nên hai dòng không so sánh với nhau.
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

    Mỗi quy tắc chỉ áp cho kết quả cùng dòng đánh số phiên bản với nó.
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

    Dấu thập phân là dấu phẩy như trong báo cáo. Dùng `tgn_nids.utils.format`
    để giá trị rất nhỏ hoặc rất gần 100% không bị làm tròn thành 0.00% hay 100.00%.
    """
    return format_percent(value, decimals=digits)


def so_nguyen(value):
    """Số đếm, dấu phân cách nghìn theo quy ước tiếng Việt là dấu chấm.

    Không dùng `f"{value:,}"`, vì dấu phẩy ở đây đã là dấu thập phân.
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

    Tỷ lệ mẫu dương lấy từ chính tập kiểm thử. Trả về (None, None) khi tệp kết
    quả thiếu số mẫu hai lớp.
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
        # Bỏ qua thư mục lưu trữ kết quả cũ (tên chứa _archive).
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
    out.append(
        "Mọi lần chạy dùng hạt giống 42, GPU Tesla T4 trên Kaggle. Giao thức "
        "`chronological_80_10_10` sắp luồng theo `FLOW_START_MILLISECONDS` rồi "
        "cắt liên tục 80/10/10. Giao thức `graphids` chia phân tầng theo cột "
        "`Attack` với hạt giống 42, cũng theo tỷ lệ 80/10/10. Ở giai đoạn một "
        "(`self_supervised`), luồng tấn công bị loại khỏi tập train nên cột Train "
        "chỉ đếm luồng Benign. Mọi chỉ số bên dưới đo trên tập test ghi ở cột "
        "Test, bộ tiền xử lý khớp trên tập train của chính lần chạy đó.\n")
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
        out.append(
            "Điều kiện đo: điểm bất thường là sai số tái thiết của giai đoạn "
            "một. Ngưỡng chọn trên tập validation sao cho Macro F1 hai lớp lớn "
            "nhất, rồi áp nguyên lên tập test. PR-AUC là average precision, "
            "ROC-AUC tính trên điểm bất thường, hai chỉ số này không phụ thuộc "
            "ngưỡng. FAR là tỷ lệ luồng Benign bị gán tấn công.\n")
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
        # Số lớp khác nhau giữa các cấu hình nên đọc từ dữ liệu, không ghi cứng.
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
        out.append(
            "Điều kiện đo: nhãn dự đoán là lớp có xác suất lớn nhất (argmax), "
            "không dùng ngưỡng. Encoder của giai đoạn một được đóng băng, chỉ "
            "classification head được huấn luyện trên tập train có nhãn.\n")
        out.append("| Cấu hình | " + " | ".join(MULTICLASS_METRICS)
                   + " | recall lớp tấn công (nhị phân quy đổi) |")
        out.append("|---" * (len(MULTICLASS_METRICS) + 2) + "|")
        for _, r in multi:
            values = " | ".join(percent(r["metrics"].get(m))
                                for m in MULTICLASS_METRICS)
            # Lấy từ binary_equivalent, không phải recall_macro.
            recall_tan_cong = r["metrics"].get("binary_equivalent", {}).get("recall")
            out.append(f"| `{r['config']['name']}` | {values} "
                       f"| {percent(recall_tan_cong)} |")

        for _, r in multi:
            out.append(f"\n**Chi tiết từng lớp, `{r['config']['name']}`**\n")
            out.append("| Lớp | Precision | Recall | F1 | FAR | Support |")
            out.append("|---|---:|---:|---:|---:|---:|")
            # Cùng đơn vị phần trăm với các bảng khác.
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
        "**Tài liệu này được sinh tự động** bởi `scripts/analysis/tong_hop_ket_qua.py`,",
        "đọc thẳng từ các tệp `result_*.json` của lần chạy thật. Không chép tay con",
        "số nào, nên tài liệu luôn khớp dữ liệu gốc. Chạy lại script sau mỗi",
        "notebook để cập nhật.",
        "",
        "Các phép đo không sinh ra từ notebook nằm ở cuối, trong mục *Các phép đo",
        "bổ trợ*. Mục đó cũng sinh tự động, đọc từ `models/supplementary/supplementary_measurements.json`",
        "do `scripts/analysis/supplementary_measurements.py` tạo ra và từ bảng tổng hợp thí nghiệm",
        "cắt bỏ. Riêng số liệu của các nghiên cứu đối sánh là trích dẫn, ghi kèm bảng",
        "và điều kiện đo của bài báo gốc. Quy tắc chọn ngưỡng và bảng tham số đã chốt",
        "xem `docs/SO_LIEU_DAU_RA_CHI_TIET.md`.",
        "",
        f"Số thí nghiệm đã có kết quả: **{sum(len(v) for v in grouped.values())}**.",
    ]
    for name in sorted(grouped):
        entries = sorted(grouped[name], key=lambda item: item[1]["config"]["name"])
        write_run_section(out, name, entries)
        write_seed_summary(out, entries)
    return "\n".join(out) + "\n"


SUPPLEMENT_PATH = os.path.join(PROJECT_ROOT, "models", "supplementary", "supplementary_measurements.json")
ABLATION_PATH = os.path.join(PROJECT_ROOT, "models", "saved", "twoDTS_ablation", "08_tong_hop",
                             "tables", "ablation_summary.csv")

# Số liệu của các nghiên cứu đối sánh là trích dẫn, không đo được, nên đây là
# phần duy nhất ghi tay. Mỗi dòng ghi kèm bảng gốc và điều kiện đo của bài báo.
PUBLISHED = [
    ("GraphIDS [10], Bảng 3", "NF-UNSW-NB15-v3", "hai lớp",
     "Macro F1 99,61% ± 0,84% · PR-AUC 99,98% ± 0,07%",
     "phân tầng ngẫu nhiên 80/10/10, trung bình nhiều hạt giống"),
    ("GraphIDS [10], Bảng 3", "NF-UNSW-NB15-v2", "hai lớp",
     "Macro F1 92,64% ± 2,17% · PR-AUC 81,16% ± 3,67%", "như trên"),
    ("GraphIDS [10], Bảng 3", "NF-CSE-CIC-IDS2018-v3", "hai lớp",
     "Macro F1 94,47% ± 2,13% · PR-AUC 88,19% ± 3,47%", "như trên, toàn bộ dữ liệu"),
    ("Anomal-E, do GraphIDS [10] đo lại, Bảng 3", "NF-UNSW-NB15-v3", "hai lớp",
     "Macro F1 94,59% ± 0,09% · PR-AUC 90,32% ± 0,41%", "như GraphIDS, biến thể tốt nhất theo PR-AUC"),
    ("Anomal-E, do GraphIDS [10] đo lại, Bảng 3", "NF-UNSW-NB15-v2", "hai lớp",
     "Macro F1 91,56% ± 2,17% · PR-AUC 74,89% ± 0,74%", "như trên"),
    ("Anomal-E, do GraphIDS [10] đo lại, Bảng 3", "NF-CSE-CIC-IDS2018-v3", "hai lớp",
     "Macro F1 67,09% ± 3,94% · PR-AUC 25,55% ± 3,83%", "như trên, toàn bộ dữ liệu"),
    ("TE-G-SAGE [14]", "NF-UNSW-NB15-v3", "hai lớp",
     "P 99,06% · R 99,99% · F1 99,52% · FAR 0,09%", "chia theo thời gian 60/30/10"),
    ("TE-G-SAGE [14], Bảng 7", "NF-UNSW-NB15-v3", "nhiều lớp",
     "Acc 95,586% · P 49,419% · R 62,738% · Macro F1 49,057% · FAR 0,446%", "chia theo thời gian 60/30/10"),
    ("GCN, trong TE-G-SAGE [14], Bảng 7", "NF-UNSW-NB15-v3", "nhiều lớp",
     "Acc 97,257% · P 39,587% · R 39,477% · Macro F1 38,781% · FAR 0,301%", "như trên"),
    ("XGBoost, trong TE-G-SAGE [14], Bảng 7", "NF-UNSW-NB15-v3", "nhiều lớp",
     "Acc 97,338% · P 68,957% · R 55,742% · Macro F1 56,782% · FAR 0,273%", "như trên"),
    ("Anomal-E-IF, bài gốc [4], Bảng 4 và 8", "NF-UNSW-NB15-v2", "hai lớp",
     "Acc 98,66% · Macro F1 92,35% · DR 98,77%",
     "giao thức riêng của Anomal-E, 4% nhiễm tấn công trong tập huấn luyện, báo cáo không dùng"),
]


def vn(value, digits):
    """Số thập phân kiểu Việt Nam, dùng cho những chỉ số không phải phần trăm."""
    return f"{value:.{digits}f}".replace(".", ",")


def manual_section():
    """Mục các phép đo bổ trợ, dựng từ tệp kết quả đã lưu thay vì chép tay."""
    import csv
    out = ["", "---", "", "# Các phép đo bổ trợ", "",
           "Những con số dưới đây không sinh ra từ notebook huấn luyện. Mục 1 và 2 "
           "đọc từ `models/supplementary/supplementary_measurements.json`, tệp do "
           "`scripts/analysis/supplementary_measurements.py` sinh ra khi chạy trên CPU với "
           "dữ liệu gốc. Mục 3 đọc từ bảng tổng hợp thí nghiệm cắt bỏ. Mục 4 là "
           "số liệu trích từ các bài báo. Mỗi mục ghi điều kiện đo ngay đầu mục.", ""]
    if not os.path.exists(SUPPLEMENT_PATH):
        out += ["> Chưa có `models/supplementary/supplementary_measurements.json`. Chạy "
                "`python scripts/analysis/supplementary_measurements.py` trước.", ""]
    else:
        with open(SUPPLEMENT_PATH, encoding="utf-8") as handle:
            sup = json.load(handle)
        loi_tat, moc = sup["data_shortcuts"], sup["tabular_baselines"]
        dia_chi = ", ".join(f"`{a}`" for a in loi_tat["attack_only_sources"])
        out += ["## 1. Lối tắt trong dữ liệu NF-UNSW-NB15-v3", "",
                f"**Điều kiện đo.** {loi_tat['condition']} Báo cáo dùng ở mục 4.9.1.", "",
                "| Phép đo | Kết quả |", "|---|---|",
                f"| Số luồng | {so_nguyen(loi_tat['n_flows'])} |",
                f"| Địa chỉ nguồn chỉ sinh lưu lượng tấn công | "
                f"{len(loi_tat['attack_only_sources'])} địa chỉ: {dia_chi} |",
                f"| Địa chỉ nguồn chỉ sinh lưu lượng Benign | {loi_tat['n_benign_only_sources']} |",
                f"| Địa chỉ nguồn sinh cả hai loại | **{loi_tat['n_mixed_sources']}** |",
                f"| `MAX_TTL` phổ biến nhất, lớp Benign | {loi_tat['max_ttl_mode_benign']} |",
                f"| `MAX_TTL` phổ biến nhất, lớp tấn công | {loi_tat['max_ttl_mode_attack']} |",
                f"| ROC-AUC khi dùng riêng `MAX_TTL` | {vn(loi_tat['max_ttl_roc_auc'], 4)} |",
                "",
                "## 2. Mốc tham chiếu dạng bảng, không dùng đồ thị", "",
                f"**Điều kiện đo.** {moc['condition']} Bộ thuộc tính gồm {moc['n_features']} cột, "
                f"scikit-learn {sup['scikit_learn']}. Báo cáo dùng ở bảng 4.24, làm tròn hai chữ số "
                "thập phân cho F1 và PR-AUC, bốn chữ số cho FAR.", "",
                "| Tập | Số luồng | Số luồng tấn công |", "|---|---:|---:|"]
        for name, part in moc["splits"].items():
            out.append(f"| {name} | {so_nguyen(part['n_flows'])} | {so_nguyen(part['n_attacks'])} |")
        out += ["", "| Mô hình | Cấu hình | F1 lớp tấn công | FAR | PR-AUC |", "|---|---|---:|---:|---:|"]
        for key, title in (("decision_tree_depth3_all_features", "Cây quyết định sâu 3, đủ thuộc tính"),
                           ("decision_tree_depth3_without_ttl", "Cây quyết định sâu 3, bỏ `MIN_TTL` và `MAX_TTL`"),
                           ("logistic_regression_all_features", "Hồi quy logistic, đủ thuộc tính")):
            row = moc[key]
            out.append(f"| {title} | `{row['model']}` | {percent(row['attack_f1'])} "
                       f"| {percent(row['far'], 4)} | {percent(row['pr_auc'])} |")
        out += ["", f"Thời gian chạy script: {vn(sup['runtime_seconds'], 1)} giây.", ""]

    out += ["## 3. Thí nghiệm cắt bỏ kiến trúc", "",
            "**Điều kiện đo.** Bốn cấu hình huấn luyện lại từ đầu qua cả hai giai đoạn trên "
            "NF-UNSW-NB15-v3, cùng một phiên, cùng hạt giống 42, giao thức `chronological_80_10_10`. "
            "Macro F1 đo trên bài toán nhiều lớp. Recall, F1 và FAR của lớp tấn công đo trên bài toán "
            "hai lớp quy đổi. Mỗi cấu hình chạy **một lần**, nên chênh lệch dưới 1 điểm chưa tách khỏi "
            "dao động giữa các lần chạy. Nguồn: `models/saved/twoDTS_ablation/08_tong_hop/tables/"
            "ablation_summary.csv`. Báo cáo dùng ở mục 4.6.", "",
            "| Cấu hình | Chiều cạnh | Macro F1 | Chênh lệch Macro F1 | Recall tấn công | F1 tấn công | FAR |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    with open(ABLATION_PATH, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            delta = vn(float(row["delta_macro_f1_pp"]), 3)
            out.append(f"| `{row['variant']}` | {row['edge_dim']} | {percent(float(row['f1_macro']))} "
                       f"| {delta} điểm | {percent(float(row['attack_recall']))} "
                       f"| {percent(float(row['attack_f1']))} | {percent(float(row['far']))} |")
    out += ["", "Phép triệt tiêu từng thuộc tính ở mục 4.7.1 là công cụ giải thích, không phải "
            "thí nghiệm cắt bỏ kiến trúc.", "",
            "## 4. Số liệu do các nghiên cứu đối sánh công bố", "",
            "**Điều kiện đo.** Trích nguyên từ bản gốc trong `references/`, không đo lại. Số hiệu "
            "tài liệu theo danh mục của báo cáo.", "",
            "| Nguồn | Bộ dữ liệu | Bài toán | Chỉ số | Điều kiện đo của bài báo |", "|---|---|---|---|---|"]
    for row in PUBLISHED:
        out.append("| " + " | ".join(row) + " |")
    out += ["", "TE-G-SAGE theo lớp, bài toán nhiều lớp, F1: Analysis 0,305 · Backdoor 0,071 · "
            "Benign 1,000 · DoS 0,260 · Exploits 0,613 · Fuzzers 0,501 · Generic 0,796 · "
            "Reconnaissance 0,579 · Shellcode 0,221 · Worms 0,500.", "",
            "TCG-IDS chưa có bản gốc nên **không trích số nào**.", ""]
    return "\n".join(out)




if __name__ == "__main__":
    document = build_document() + manual_section()
    # newline="\n" để tệp sinh ra giống nhau trên Windows và macOS.
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    print(f"Đã ghi: {OUTPUT_PATH}")
    print(f"Dung lượng: {len(document) / 1024:.1f} KB, "
          f"{document.count(chr(10))} dòng")
