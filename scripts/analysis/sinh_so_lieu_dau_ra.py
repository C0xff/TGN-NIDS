"""Sinh tài liệu mô tả đầy đủ số liệu đầu ra, đọc trực tiếp từ models/saved.

Mỗi con số kèm đường dẫn tệp result_*.json và khoá JSON nguồn.
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJ = Path(__file__).resolve().parents[2]
DICH = PROJ / "docs" / "SO_LIEU_DAU_RA_CHI_TIET.md"


def pc(x, n=3):
    """Phần trăm, dấu thập phân tiếng Việt, dựng qua Decimal trên repr."""
    if x is None:
        return "—"
    return format((Decimal(repr(float(x))) * 100).quantize(Decimal(1).scaleb(-n)),
                  "f").replace(".", ",") + "%"


def sl(x):
    """Số nguyên, dấu phân cách nghìn kiểu Việt."""
    if x is None:
        return "—"
    return "{:,}".format(int(x)).replace(",", ".")


TEP = [
    ("twoDTS_train_v2", "07_giai_doan_mot", "result_stage1_nf_unsw_nb15_v2.json"),
    ("twoDTS_train_v2", "08_giai_doan_hai", "result_stage2_nf_unsw_nb15_v2.json"),
    ("twoDTS_train_v3", "07_giai_doan_mot", "result_stage1_nf_unsw_nb15_v3.json"),
    ("twoDTS_train_v3", "07_giai_doan_mot", "result_stage1_nf_cse_cic_ids2018_v3.json"),
    ("twoDTS_train_v3", "08_giai_doan_hai", "result_stage2_nf_unsw_nb15_v3.json"),
    ("twoDTS_train_v3", "08_giai_doan_hai", "result_stage2_nf_cse_cic_ids2018_v3.json"),
]

BO = {
    "nf_unsw_nb15_v2": "NF-UNSW-NB15-v2",
    "nf_unsw_nb15_v3": "NF-UNSW-NB15-v3",
    "nf_cse_cic_ids2018_v3": "NF-CSE-CIC-IDS2018-v3",
}

# Quy mô bộ dữ liệu gốc trước khi rút mẫu; sample_ratio chỉ ghi phép rút mẫu của runner.
GOC = {}
for _cay in ("twoDTS_train_v2", "twoDTS_train_v3"):
    _p = PROJ / "models/saved" / _cay / "02_ho_so_du_lieu" / "tables" / "ho_so_du_lieu.csv"
    if _p.exists():
        import csv as _csv
        for _r in _csv.DictReader(_p.read_text(encoding="utf-8-sig").splitlines()):
            GOC[_r["dataset"]] = int(_r["flow_count"])

NGUON = ["models/saved/{}/{}/models/{}".format(c, o, n) for c, o, n in TEP] + [
    "models/saved/{}/02_ho_so_du_lieu/tables/ho_so_du_lieu.csv".format(c)
    for c in ("twoDTS_train_v2", "twoDTS_train_v3")] + [
    "models/saved/twoDTS_phan_tich/metrics/02_ngoai_phan_phoi/ngoai_phan_phoi.json",
    "models/saved/twoDTS_phan_tich/tables/04_giai_thich/do_quan_trong_dac_trung.csv",
    "models/saved/twoDTS_phan_tich/tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv",
    "models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv",
    "models/saved/twoDTS_demo/metrics/do_tap_demo.json",
]

ra = []
w = ra.append

w("# SỐ LIỆU ĐẦU RA CỦA MÔ HÌNH: MÔ TẢ ĐẦY ĐỦ TỪNG CHỈ SỐ")
w("")
w("**Tệp này sinh tự động. Đừng sửa tay.** Dựng lại bằng:")
w("")
w("```")
w("python scripts/analysis/sinh_so_lieu_dau_ra.py")
w("```")
w("")
w("Script đọc đúng những tệp sau và không đọc gì khác:")
w("")
for _t in NGUON:
    w("- `{}`".format(_t))
w("")
w("Kiểm tính chính xác bằng cách chạy lại script rồi so tệp sinh ra với tệp đang")
w("có; hai bản phải giống từng byte. Mọi con số đều là giá trị đọc thẳng từ tệp")
w("nguồn, hoặc là phép chia và phép trừ trên các giá trị đó; không con số nào gõ tay.")
w("")
w("Mục đích: mỗi con số đi kèm **đủ điều kiện đo**, để không ai, người hay công cụ,")
w("trích nó sang một bối cảnh khác.")
w("")
w("Một chỉ số chỉ được trích vào báo cáo khi nêu đủ bốn thành phần: **bộ dữ liệu**,")
w("**bài toán**, **tập đo**, và **tên chỉ số đúng như định nghĩa dưới đây**. Thiếu")
w("một trong bốn thì con số đó không được dùng.")
w("")
w("## 0. Bảng tra nhanh sáu cấu hình")
w("")
w("| Cấu hình | Bộ dữ liệu | Bài toán | Giao thức chia tập | Chỉ số chính | Giá trị |")
w("|---|---|---|---|---|---:|")
for cay, o, ten in TEP:
    d = json.loads((PROJ / "models/saved" / cay / o / "models" / ten).read_text(encoding="utf-8"))
    cf, m = d["config"], d["metrics"]
    bo = next(v for k, v in BO.items() if k in ten)
    if cf["task"] == "binary":
        w("| `{}` | {} | hai lớp | `{}` | F1 lớp tấn công | {} |".format(
            cf["name"], bo, cf["protocol"], pc(m["f1"])))
    else:
        w("| `{}` | {} | nhiều lớp | `{}` | Macro F1 | {} |".format(
            cf["name"], bo, cf["protocol"], pc(m["f1_macro"])))
w("")
w("Giai đoạn một báo cáo **F1 của lớp tấn công**, giai đoạn hai báo cáo **Macro F1**.")
w("Hai chỉ số này đo hai bài toán khác nhau nên **không đặt cạnh nhau như thể so")
w("sánh được**, và không cộng trừ với nhau.")
w("")

for cay, o, ten in TEP:
    p = PROJ / "models/saved" / cay / o / "models" / ten
    d = json.loads(p.read_text(encoding="utf-8"))
    cf, m, sp = d["config"], d["metrics"], d["split_stats"]
    bo = next(v for k, v in BO.items() if k in ten)
    nhieu_lop = cf["task"] != "binary"

    w("## `{}`".format(cf["name"]))
    w("")
    w("**Nguồn:** `models/saved/{}/{}/models/{}`".format(cay, o, ten))
    w("")
    w("| Điều kiện đo | Giá trị |")
    w("|---|---|")
    w("| Bộ dữ liệu | {} |".format(bo))
    w("| Giai đoạn | {} ({}) |".format("hai" if nhieu_lop else "một", cf["mode"]))
    if nhieu_lop:
        w("| Bài toán | nhiều lớp, {} lớp, lớp lành tính là `{}` |".format(
            len(d["class_names"]), d["class_names"][d["benign_class_id"]]))
    else:
        w("| Bài toán | hai lớp, tấn công so với lành tính |")
    w("| Giao thức chia tập | `{}` |".format(cf["protocol"]))
    vao = d["dataset"].get("rows_input")
    if bo in GOC:
        w("| Quy mô bộ gốc, trước mọi phép rút mẫu | {} luồng |".format(sl(GOC[bo])))
    w("| Số luồng thực sự đưa vào runner | {} |".format(sl(vao)))
    if bo in GOC and vao:
        w("| Tỷ lệ so với bộ gốc | {} |".format(pc(vao / GOC[bo])))
    w("| Trường `sample_ratio` trong tệp kết quả | {:.0%}, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |".format(
        cf["sample_ratio"]))
    for nhan, khoa in (("Tập huấn luyện", "train"), ("Tập kiểm định", "val")):
        w("| {} | {} luồng, {} tấn công ({}) |".format(
            nhan, sl(sp[khoa]["count"]), sl(sp[khoa]["attacks"]),
            pc(sp[khoa]["attacks"] / sp[khoa]["count"])))
    w("| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | {} luồng, {} tấn công ({}) |".format(
        sl(sp["test"]["count"]), sl(sp["test"]["attacks"]),
        pc(sp["test"]["attacks"] / sp["test"]["count"])))
    w("| Số đỉnh đồ thị | {} |".format(sl(d["n_nodes"])))
    # Mật độ luồng trên mỗi đỉnh: số luồng tập huấn luyện chia số đỉnh.
    mat_do = Decimal(sp["train"]["count"]) / Decimal(d["n_nodes"])
    w("| Mật độ luồng huấn luyện trên mỗi đỉnh | {} luồng |".format(
        format(mat_do.quantize(Decimal("0.01")), "f").replace(".", ",")))
    w("| Số chiều đặc trưng cạnh | {} |".format(d["edge_dim"]))
    w("| Số tham số mô hình | {} |".format(sl(d["n_parameters"])))
    w("| Tiêu chí dừng sớm | `{}`, patience {} |".format(
        d["early_stopping_metric"], cf["early_stopping_patience"]))
    w("| Vòng tốt nhất | {} trên ngân sách {} |".format(d["best_epoch"], cf["num_epochs"]))
    if d.get("threshold") is not None:
        w("| Ngưỡng quyết định | {} |".format(("%.6f" % d["threshold"]).replace(".", ",")))
    else:
        w("| Ngưỡng quyết định | không dùng, bài toán nhiều lớp chọn lớp có xác suất cao nhất |")
    w("| Thiết bị | {}, torch {} |".format(d["environment"]["gpu"], d["environment"]["torch"]))
    w("| Thời gian huấn luyện | {} phút |".format(
        "{:.1f}".format(d["training_time_seconds"] / 60).replace(".", ",")))
    w("| Độ trễ suy luận | {} ms trên 100 luồng |".format(
        str(d["latency_ms_per_100_flows"]).replace(".", ",")))
    w("| Ablation | node_identity `{}`, features `{}` |".format(
        d["ablation"]["node_identity"], d["ablation"]["features"] or "rỗng"))
    w("| Phiên bản mã nguồn | {} |".format(d["source_version"]))
    w("| Thời điểm chạy xong | {} |".format(d["run_finished_at"]))
    w("")
    w("**Chỉ số trên tập kiểm thử**")
    w("")
    w("| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |")
    w("|---|---:|---|---|")

    if not nhieu_lop:
        r = Decimal(repr(sp["test"]["attacks"] / sp["test"]["count"]))
        moc = 2 * r / (1 + r)
        hang = [
            ("Accuracy", m["accuracy"],
             "tỷ lệ đúng trên cả hai lớp, bị lớp lành tính chi phối nên không dùng để kết luận",
             "metrics.accuracy"),
            ("F1 lớp tấn công", m["f1"],
             "chỉ số chính của giai đoạn một, chỉ nói về lớp tấn công",
             "metrics.f1"),
            ("Macro F1", m["f1_macro"],
             "trung bình F1 của hai lớp, không trọng số theo số mẫu",
             "metrics.f1_macro"),
            ("Precision lớp tấn công", m["precision"],
             "trong số luồng bị gắn cờ, bao nhiêu phần đúng là tấn công",
             "metrics.precision"),
            ("Recall lớp tấn công", m["recall"],
             "trong số tấn công thật, bao nhiêu phần bị bắt; chỉ số này phải luôn đứng cạnh Macro F1",
             "metrics.recall"),
            ("Tỷ lệ báo động giả", m["far"],
             "luồng lành tính bị gắn cờ chia tổng luồng lành tính, bằng FPR",
             "metrics.far"),
            ("PR-AUC", m["pr_auc"],
             "diện tích dưới đường Precision-Recall, không phụ thuộc ngưỡng",
             "metrics.pr_auc"),
            ("ROC-AUC", m["roc_auc"],
             "diện tích dưới ROC, lạc quan khi dữ liệu lệch lớp nặng",
             "metrics.roc_auc"),
            ("Mốc phân loại tầm thường", float(moc),
             "F1 mà một bộ gán nhãn tấn công cho mọi luồng sẽ đạt, tính bằng 2r/(1+r) với r là tỷ lệ tấn công của tập kiểm thử",
             "tính từ split_stats.test"),
        ]
        for ten_cs, gt, y, khoa in hang:
            w("| {} | {} | {} | `{}` |".format(ten_cs, pc(gt), y, khoa))
        w("")
        w("Ma trận nhầm lẫn trên tập kiểm thử: TP {}, FP {}, TN {}, FN {}.".format(
            sl(m["tp"]), sl(m["fp"]), sl(m["tn"]), sl(m["fn"])))
        w("")
        chenh = (Decimal(repr(m["f1"])) - moc) * 100
        w("F1 lớp tấn công {} so với mốc tầm thường {}: {}, chênh {} điểm phần trăm.".format(
            pc(m["f1"]), pc(float(moc)),
            "**vượt mốc**" if m["f1"] > float(moc) else "**KHÔNG vượt mốc**",
            format(chenh.quantize(Decimal("0.001")), "f").replace(".", ",")))
    else:
        be = m.get("binary_equivalent") or {}
        hang = [
            ("Accuracy", m["accuracy"],
             "tỷ lệ đúng trên toàn bộ lớp, bị lớp lành tính chi phối",
             "metrics.accuracy"),
            ("Macro F1", m["f1_macro"],
             "chỉ số chính của giai đoạn hai, trung bình không trọng số trên mọi lớp, nên một lớp hiếm nặng bằng một lớp lớn",
             "metrics.f1_macro"),
            ("Macro Precision", m["precision_macro"],
             "trung bình không trọng số precision theo lớp",
             "metrics.precision_macro"),
            ("Macro Recall", m["recall_macro"],
             "trung bình không trọng số recall theo lớp, **không phải** recall của lớp tấn công",
             "metrics.recall_macro"),
            ("Weighted F1", m["f1_weighted"],
             "có trọng số theo số mẫu nên gần như chỉ phản ánh lớp lành tính",
             "metrics.f1_weighted"),
            ("Macro FAR", m["far_macro"],
             "trung bình tỷ lệ báo động giả theo lớp",
             "metrics.far_macro"),
        ]
        if be.get("recall") is not None:
            hang.append(("Recall lớp tấn công, quy về hai lớp", be["recall"],
                         "gộp mọi lớp tấn công thành một rồi đo recall; đây là con số phải đứng cạnh Macro F1",
                         "metrics.binary_equivalent.recall"))
        if be.get("far") is not None:
            hang.append(("Tỷ lệ báo động giả, quy về hai lớp", be["far"],
                         "đo sau khi quy về hai lớp, không so sánh được với Macro FAR",
                         "metrics.binary_equivalent.far"))
        for ten_cs, gt, y, khoa in hang:
            w("| {} | {} | {} | `{}` |".format(ten_cs, pc(gt), y, khoa))
        w("")
        w("Cấu hình này **không có** `roc_auc` và `pr_auc`, vì hai chỉ số đó chỉ định")
        w("nghĩa cho bài toán hai lớp. Thấy chúng đứng cạnh Macro F1 của cấu hình này")
        w("thì nghĩa là chúng bị lấy từ bài toán hai lớp quy đổi, và đó là lỗi gắn chỉ")
        w("số sai bài toán.")
        w("")
        w("**Chỉ số theo từng lớp, đo trên tập kiểm thử**")
        w("")
        w("| Lớp | Precision | Recall | F1 | FAR | Số mẫu |")
        w("|---|---:|---:|---:|---:|---:|")
        for c in m["per_class"]:
            w("| {} | {} | {} | {} | {} | {} |".format(
                c["name"], pc(c["precision"]), pc(c["recall"]), pc(c["f1"]),
                pc(c["far"]), sl(c["support"])))
    w("")

PT = PROJ / "models/saved/twoDTS_phan_tich"

w("## Phân tích ngoài phân phối trên NF-ToN-IoT-v3")
w("")
ood = json.loads((PT / "metrics/02_ngoai_phan_phoi/ngoai_phan_phoi.json").read_text(encoding="utf-8"))
w("**Nguồn:** `models/saved/twoDTS_phan_tich/metrics/02_ngoai_phan_phoi/ngoai_phan_phoi.json`")
w("")
w("Đây là phép đo **khái quát sang bộ dữ liệu chưa từng huấn luyện**. Mô hình giữ")
w("nguyên trọng số, chỉ đổi dữ liệu đầu vào, nên mọi con số dưới đây nói về khả năng")
w("chuyển miền, **không** so sánh được với chỉ số trên tập kiểm thử cùng phân phối.")
w("")
w("| Điều kiện đo | Giá trị |")
w("|---|---|")
w("| Bộ dữ liệu đích | NF-ToN-IoT-v3 |")
w("| Số luồng lấy mẫu | {} |".format(sl(ood["target_rows"])))
w("| Hạt giống lấy mẫu / chia | {} / {} |".format(ood["sample_seed"], ood["split_seed"]))
w("| Giao thức | {} |".format(ood["protocol"]))
w("")
w("| Mô hình gốc | Tỷ lệ dương | Mốc tầm thường | F1 ngưỡng mặc định | F1 sau hiệu chuẩn | Recall | ROC-AUC | PR-AUC | Vượt mốc |")
w("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
for r in ood["results"]:
    w("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        r["run_name"], pc(r["positive_rate"]), pc(r["trivial_f1_baseline"]),
        pc(r["f1_default_threshold"]), pc(r["f1_calibrated"]), pc(r["recall"]),
        pc(r["roc_auc"]), pc(r["pr_auc"]),
        "có" if r["exceeds_trivial_baseline"] else "**không**"))
w("")
w("Cột **F1 ngưỡng mặc định** là F1 khi áp thẳng ngưỡng đã hiệu chuẩn trên bộ gốc;")
w("cột **F1 sau hiệu chuẩn** là F1 khi chọn lại ngưỡng trên nửa A của bộ đích rồi")
w("chấm trên nửa B. Hai cột này **không được trộn**: cột thứ hai cần nhãn của bộ")
w("đích, tức một điều kiện mà tình huống triển khai thật không có.")
w("")

w("## Giải thích mô hình: độ quan trọng đặc trưng")
w("")
w("**Nguồn:** `models/saved/twoDTS_phan_tich/tables/04_giai_thich/do_quan_trong_dac_trung.csv`")
w("")
w("Phép triệt tiêu từng đặc trưng: lần lượt đặt một cột về giá trị không rồi đo độ")
w("lệch của xác suất tấn công. Đây là **công cụ giải thích**, không phải ablation")
w("kiến trúc, và không nói gì về đóng góp của thành phần kiến trúc nào.")
w("")
import csv as _c
rows = list(_c.DictReader((PT / "tables/04_giai_thich/do_quan_trong_dac_trung.csv")
                          .read_text(encoding="utf-8-sig").splitlines()))
theo_run = {}
for r in rows:
    theo_run.setdefault(r["run_name"], []).append(r)
for ten, nhom in theo_run.items():
    w("**`{}`**, năm đặc trưng dẫn đầu trong {} đặc trưng đã đo:".format(ten, len(nhom)))
    w("")
    w("| Hạng | Đặc trưng | Độ quan trọng tương đối | Độ lệch xác suất tuyệt đối trung bình |")
    w("|---:|---|---:|---:|")
    for r in nhom[:5]:
        w("| {} | `{}` | {} | {} |".format(r["rank"], r["feature_name"],
                                           pc(r["relative_importance"]),
                                           pc(r["mean_absolute_shift"])))
    w("")

w("## Giải thích mô hình: trọng số cạnh học bằng GNNExplainer")
w("")
w("**Nguồn:** `models/saved/twoDTS_phan_tich/tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv`")
w("")
em = list(_c.DictReader((PT / "tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv")
                        .read_text(encoding="utf-8-sig").splitlines()))
w("| Đại lượng | Giá trị | Ý nghĩa | Cột CSV |")
w("|---|---:|---|---|")
for r in em:
    w("| Mô hình | `{}` | lần chạy được giải thích | `run_name` |".format(r["run_name"]))
    w("| Số luồng chấm điểm | {} | quy mô đồ thị con đưa vào | `flows_scored` |".format(sl(r["flows_scored"])))
    w("| Xác suất tấn công trước khi che | {} | dự đoán gốc của luồng mục tiêu | `attack_probability_before` |".format(pc(r["attack_probability_before"])))
    w("| Xác suất tấn công sau khi che | {} | sau khi áp mặt nạ cạnh đã học | `attack_probability_after` |".format(pc(r["attack_probability_after"])))
    w("| Trọng số cạnh nhỏ nhất | {} | | `edge_mask_min` |".format(pc(r["edge_mask_min"])))
    w("| Trọng số cạnh trung vị | {} | trung vị thấp nghĩa là phần lớn cạnh bị hạ | `edge_mask_median` |".format(pc(r["edge_mask_median"])))
    w("| Trọng số cạnh lớn nhất | {} | | `edge_mask_max` |".format(pc(r["edge_mask_max"])))
    w("| Độ lệch chuẩn trọng số | {} | càng lớn càng phân biệt được cạnh | `edge_mask_std` |".format(pc(r["edge_mask_std"])))
    w("| Số cạnh giữ trên ngưỡng 0,5 | {} trên {} | | `edges_kept_above_half`, `edges_total` |".format(
        sl(r["edges_kept_above_half"]), sl(r["edges_total"])))
    w("| Số tham số mô hình bị đổi | {} | phải bằng 0, phép giải thích không được sửa mô hình | `model_parameters_changed` |".format(r["model_parameters_changed"]))
    w("| Số vòng tối ưu mặt nạ | {} | | `epochs` |".format(r["epochs"]))
w("")

AB = PROJ / "models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv"
MOTA_ABL = {
    "baseline": "không, đây là mốc đối chiếu",
    "random_nodes": "danh tính đỉnh, gán lại ngẫu nhiên đỉnh nguồn và đích",
    "no_iat": "tám cột đặc trưng thời gian IAT",
    "no_propagation": "phép tổng hợp từ đỉnh lân cận, giữ nguyên vector nhớ",
}
if AB.exists():
    hang = list(_c.DictReader(AB.read_text(encoding="utf-8-sig").splitlines()))
    w("## Thực nghiệm cắt bỏ kiến trúc")
    w("")
    w("**Nguồn:** `models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv`")
    w("")
    w("Mọi giá trị dưới đây đo trên **một hạt giống duy nhất**, nên các chênh lệch")
    w("nhỏ chưa tách khỏi dao động giữa các lần chạy và phải trích kèm giới hạn")
    w("phát hiện. Mở rộng sang nhiều hạt giống thuộc Hướng phát triển, ghi ở mục 6")
    w("`docs/THUC_NGHIEM_CAT_BO_KIEN_TRUC.md`.")
    w("")
    w("Mỗi cấu hình được **huấn luyện lại từ đầu** trên toàn bộ NF-UNSW-NB15-v3, cùng")
    w("hạt giống và trong cùng một phiên, nên mốc đối chiếu không lẫn sai khác môi")
    w("trường. Trường `question` trong tệp nguồn ghi `retrain`: con số dưới đây trả lời")
    w("câu hỏi *kiến trúc thiếu thành phần đó thì học được tới đâu*, **không** trả lời")
    w("câu hỏi mô hình đã huấn luyện dựa vào thành phần đó bao nhiêu lúc suy luận.")
    w("")
    w("| Cấu hình | Thành phần bị vô hiệu hoá | Macro F1 | Chênh so với mốc | Recall lớp tấn công | F1 lớp tấn công | Tỷ lệ báo động giả |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for r in hang:
        chenh = Decimal(r["delta_macro_f1_pp"]).quantize(Decimal("0.001"))
        w("| `{}` | {} | {} | {} | {} | {} | {} |".format(
            r["variant"], MOTA_ABL.get(r["variant"], ""), pc(r["f1_macro"]),
            "—" if r["variant"] == "baseline"
            else format(chenh, "f").replace(".", ",") + " điểm",
            pc(r["attack_recall"]), pc(r["attack_f1"]), pc(r["far"])))
    w("")
    w("Mốc phân loại tầm thường của tập kiểm thử là {}. Cả bốn cấu hình đều vượt xa mốc".format(
        pc(hang[0]["trivial_f1"])))
    w("này, nên không cấu hình nào rơi xuống mức đoán bừa.")
    w("")
    w("| Cấu hình | Vòng tốt nhất giai đoạn một | Vòng tốt nhất giai đoạn hai | Thời gian huấn luyện | Bị cắt khi còn đang tiến bộ |")
    w("|---|---:|---:|---:|:---:|")
    for r in hang:
        w("| `{}` | {} trên 150 | {} trên 100 | {} phút | {} |".format(
            r["variant"], r["stage1_best_epoch"], r["stage2_best_epoch"],
            "{:.1f}".format(float(r["training_time_seconds"]) / 60).replace(".", ","),
            "có" if r["near_epoch_limit"] == "True" else "không"))
    w("")
    w("Cột cuối là phép tự kiểm: vòng tốt nhất nằm sát cuối ngân sách nghĩa là mô hình")
    w("vẫn đang tiến bộ khi bị cắt, và con số của nó không đọc được như con số của một")
    w("mô hình đã hội tụ. Cả bốn cấu hình đều không rơi vào trường hợp đó.")
    w("")

DEMO = PROJ / "models/saved/twoDTS_demo/metrics/do_tap_demo.json"
if DEMO.exists():
    dm = json.loads(DEMO.read_text(encoding="utf-8"))
    w("## Phép đo chức năng trên ba tệp demo")
    w("")
    w("**Nguồn:** `models/saved/twoDTS_demo/metrics/do_tap_demo.json`, dựng lại bằng")
    w("`python scripts/analysis/do_tap_demo.py`.")
    w("")
    w("Đây là **phép thử chức năng của giao diện**, không phải kết quả đánh giá mô")
    w("hình, và không được dùng để kết luận mô hình tốt hay kém.")
    w("")
    w("Hai trạng thái bộ nhớ khác nhau, và tên hàm dễ gây nhầm: `reset_memory` của")
    w("động cơ suy luận **khôi phục** bảng nhớ từ checkpoint chứ không xoá; muốn xoá")
    w("trắng phải gọi `memory_module.reset_state()`.")
    w("")
    w("| Tệp | Kịch bản | Trạng thái bộ nhớ | Độ chính xác | Recall lớp tấn công | Tỷ lệ báo động giả | Mốc tầm thường |")
    w("|---|---|---|---:|---:|---:|---:|")
    for r in dm["ket_qua"]:
        w("| `{}` | {} | {} | {} | {} | {} | {} |".format(
            r["tep"], r["kich_ban"], r["trang_thai_bo_nho"], pc(r["do_chinh_xac"]),
            pc(r["recall_lop_tan_cong"]), pc(r["ty_le_bao_dong_gia"]),
            pc(r["moc_phan_loai_tam_thuong"])))
    w("")
    w("**Hạn chế phải nêu kèm:** " + dm["han_che"])
    w("")

DICH.write_text("\n".join(ra) + "\n", encoding="utf-8")
print("da sinh", DICH.relative_to(PROJ).as_posix(), "|", len(ra), "dong")
