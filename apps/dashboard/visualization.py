"""Biểu đồ và thành phần trực quan cho dashboard."""
import decimal
import html
import plotly.graph_objects as go

from tgn_nids.utils.format import percent as _percent
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def _esc(value) -> str:
    # An toàn khi chèn chuỗi vào HTML.
    return html.escape("" if value is None else str(value), quote=True)


def plot_traffic_timeline(df: pd.DataFrame) -> go.Figure:
    # Phân bố luồng theo thời gian.
    df_plot = df.copy()
    # Chia cửa sổ theo thứ tự luồng.
    df_plot["flow_idx"] = range(len(df_plot))

    # Ưu tiên nhãn thật nếu có.
    df_plot["type"] = None
    if "Label" in df_plot.columns:
        df_plot["type"] = df_plot["Label"].map({0: "Benign", 1: "Attack"})
    # Dùng dự đoán khi không có nhãn phù hợp.
    if df_plot["type"].isna().all() and "prediction" in df_plot.columns:
        df_plot["type"] = df_plot["prediction"].map({0: "Benign", 1: "Attack"})
    if df_plot["type"].isna().all():
        df_plot["type"] = "Unknown"

    # Chia dữ liệu thành các cửa sổ đều nhau.
    window_size = max(1, len(df_plot) // 20)
    df_plot["window"] = df_plot["flow_idx"] // window_size

    grouped = df_plot.groupby(["window", "type"]).size().reset_index(name="count")

    fig = go.Figure()

    # Màu cho luồng bình thường và tấn công.
    colors = {"Benign": CHART["benign"], "Attack": CHART["attack"]}

    for t in ["Benign", "Attack"]:
        mask = grouped["type"] == t
        fig.add_trace(go.Bar(
            x=grouped[mask]["window"],
            y=grouped[mask]["count"],
            name=t,
            marker_color=colors.get(t, "#888"),
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="phân bố luồng theo thứ tự · lành tính so với tấn công",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        xaxis_title="Cửa sổ theo thứ tự luồng",
        yaxis_title="Number of Flows",
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=CHART["text"], family="IBM Plex Sans, sans-serif"),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
    )

    return _apply_axis_theme(fig)


def plot_attack_distribution(df: pd.DataFrame) -> go.Figure:
    # Tỷ trọng từng loại tấn công.
    # Dùng dự đoán khi không có nhãn thật.
    if "Label" in df.columns and "Attack" in df.columns:
        attack_counts = df[df["Label"] == 1]["Attack"].value_counts()
    elif "prediction" in df.columns and "prediction_label" in df.columns:
        attack_counts = df[df["prediction"] == 1]["prediction_label"].value_counts()
    else:
        attack_counts = pd.Series(dtype=int)

    if len(attack_counts) == 0:
        attack_counts = pd.Series({"No Attacks": 1})

    # Màu cho từng loại tấn công.
    colors_palette = list(CHART["cat"])

    fig = go.Figure(data=[go.Pie(
        labels=attack_counts.index,
        values=attack_counts.values,
        hole=0.55,
        marker=dict(colors=colors_palette[:len(attack_counts)], line=dict(color="#1a0f07", width=2)),
        textinfo="percent",
        # Hiển thị một chữ số thập phân.
        texttemplate="%{percent:.1%}",
        textposition="inside",
        textfont=dict(size=14, color="#ffffff", family="Arial, sans-serif"),
        insidetextfont=dict(color="#ffffff", size=11, family="Arial, sans-serif"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text="attack type distribution",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color=CHART["text"], size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )

    return _apply_axis_theme(fig)


def plot_confusion_matrix(metrics: dict, title: str = "Confusion Matrix") -> go.Figure:
    # Ma trận nhầm lẫn cho nhị phân và đa lớp.
    cm = metrics.get("confusion_matrix", None)

    # Trả hình rỗng khi không có dữ liệu.
    if cm is None:
        empty = go.Figure()
        empty.add_annotation(
            text="Không có ma trận nhầm lẫn trong kết quả của mô hình này",
            showarrow=False, font=dict(size=13, color=CHART["text"]))
        empty.update_layout(
            title=title, xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor=CHART["bg"], plot_bgcolor=CHART["bg"], height=320)
        return empty

    cm = np.array(cm)
    n_dim = cm.shape[0] if len(cm.shape) > 1 else 2

    # Đọc tên lớp hoặc gán nhãn theo số chiều.
    stored_names = metrics.get("class_names")
    if stored_names and len(stored_names) == n_dim:
        labels = [str(name) for name in stored_names]
    elif n_dim == 2:
        labels = ["Benign (0)", "Attack (1)"]
    else:
        labels = [f"Class {i}" for i in range(n_dim)]

    total = cm.sum()
    cm_pct = (cm / total * 100) if total > 0 else cm

    text_template = [[f"<b>{cm[i][j]:,.0f}</b>" if n_dim > 2 else f"<b>{cm[i][j]:,.0f}</b><br>({decimal.Decimal(repr(float(cm_pct[i][j]))).normalize():f}%)"
                       for j in range(n_dim)] for i in range(n_dim)]

    # Chọn màu chữ tương phản với ô.
    z_max = float(cm.max()) if cm.size and cm.max() > 0 else 1.0
    annotations = []
    for i in range(n_dim):
        for j in range(n_dim):
            frac = float(cm[i][j]) / z_max
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=text_template[i][j],
                showarrow=False,
                font=dict(
                    size=10 if n_dim > 2 else 16,
                    color="#ffffff" if frac > 0.55 else CHART["text"],
                    family="IBM Plex Mono, monospace",
                ),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=CHART["seq"],
        showscale=True if n_dim > 2 else False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(
            text=str(title).lower(),
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420 if n_dim > 2 else 380,
        margin=dict(l=70, r=20, t=50, b=70),
    )

    return _apply_axis_theme(fig)


# Màu cố định cho biểu đồ đối sánh.
METRICS_COMPARISON_PALETTE = [
    "#f97316", "#fbbf24", "#6b8e9f", "#dc2626", "#eab308", "#a3a3a3",
    "#c2410c", "#7d9471", "#d6d3d1", "#9a3412", "#f5a97f", "#8a8175",
    "#b45309", "#5f7d8c",
]


def plot_metrics_comparison(metrics_dict: dict) -> go.Figure:
    # So sánh chỉ số và mốc công bố.
    metric_keys = ["accuracy", "precision", "recall", "f1_macro", "fpr", "auc_roc"]
    display_names = ["Accuracy", "Precision", "Recall", "Macro F1", "FPR", "AUC-ROC"]

    fig = go.Figure()
    palette = METRICS_COMPARISON_PALETTE

    for i, (version, metrics) in enumerate(metrics_dict.items()):
        values = []
        for key in metric_keys:
            # Để trống khi không có chỉ số.
            val = metrics.get(key)
            if val is None:
                values.append(None)
            else:
                values.append(val * 100 if val <= 1.0 else val)

        fig.add_trace(go.Bar(
            name=version,
            x=display_names,
            y=values,
            marker_color=palette[i % len(palette)],
            opacity=0.9,
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.3f}%<extra></extra>",
        ))

    # Dùng chú thích HTML riêng cho biểu đồ.
    fig.update_layout(
        barmode="group",
        showlegend=False,
        title=dict(
            text="model performance comparison",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 110]),
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )

    return _apply_axis_theme(fig)


def build_metrics_comparison_legend_html(metrics_dict: dict) -> str:
    # Chú thích màu HTML cho biểu đồ đối sánh.
    palette = METRICS_COMPARISON_PALETTE
    chips = []
    for i, version in enumerate(metrics_dict.keys()):
        color = palette[i % len(palette)]
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin:4px 12px 4px 0;font-size:0.82rem;color:#e7e5e4;">'
            f'<span style="width:11px;height:11px;border-radius:3px;'
            f'background:{color};flex-shrink:0;"></span>{version}</span>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;padding:8px 4px 0 4px;">{"".join(chips)}</div>'


def plot_feature_importance(importance: dict, top_n: int = 15) -> go.Figure:
    # Biểu đồ độ quan trọng của các đặc trưng.
    # Giữ lại N đặc trưng cao nhất.
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    # Màu tăng theo độ lớn giá trị.
    colors = [f"rgba(245, 158, 11, {0.3 + 0.7 * (v / max(values)) if max(values) > 0 else 0.5})" for v in values]

    fig = go.Figure(data=go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color="rgba(245, 158, 11, 0.8)", width=1),
        ),
        text=[f"{v:.3f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=CHART["text"]),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="feature importance · perturbation analysis",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        # Chừa chỗ cho nhãn cuối cột.
        xaxis=dict(range=[0, max(values) * 1.2] if len(values) else None),
        xaxis_title="Importance (%)",
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=180, r=60, t=50, b=40),
        height=max(350, top_n * 28),
    )

    return _apply_axis_theme(fig)


def loc_canh_theo_nguong(edges: list, min_score_filter: float = 0.0) -> list:
    # Lọc luồng bình thường dưới ngưỡng điểm.
    if not min_score_filter:
        return list(edges)
    giu = []
    for e in edges:
        diem = e.get("anomaly_score")
        if (diem is not None and not e.get("is_anomaly")
                and float(diem) < min_score_filter):
            continue
        giu.append(e)
    return giu


def render_ego_network_html(
    subgraph_data: dict,
    height: str = "520px",
    enable_physics: bool = False,
) -> str:
    # Tạo đồ thị lân cận HTML bằng Pyvis.
    try:
        from pyvis.network import Network
    except ImportError:
        return ("<p style='color:#ff4757'>Chưa cài Pyvis. "
                "Chạy: pip install pyvis</p>")

    net = Network(
        height=height,
        width="100%",
        bgcolor=CHART["bg"],
        font_color=CHART["text"],
        directed=True,
        cdn_resources="in_line",
    )

    physics_json = "true" if enable_physics else "false"
    net.set_options(f"""
    {{
        "physics": {{
            "enabled": {physics_json},
            "barnesHut": {{
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.09
            }}
        }},
        "edges": {{
            "arrows": {{"to": {{"enabled": true, "scaleFactor": 1.0}}}},
            "smooth": {{"type": "curvedCW", "roundness": 0.2}},
            "font": {{
                "color": "#000000",
                "size": 11,
                "align": "middle",
                "background": "#fed7aa",
                "strokeWidth": 1,
                "strokeColor": "#ffffff"
            }}
        }},
        "nodes": {{
            "font": {{"size": 13, "color": "#f5f5f4", "face": "monospace"}}
        }}
    }}
    """)

    nodes = subgraph_data.get("nodes", [])
    edges = subgraph_data.get("edges", [])

    # Xếp đỉnh theo các vòng k-hop.
    import math as _math
    by_hop = {}
    for nd in nodes:
        by_hop.setdefault(int(nd.get("hop", 99)), []).append(nd)

    # Tính bán kính để các đỉnh không chồng lấn.
    RING_R = {0: 0}
    r_truoc = 0
    for hop in sorted(h for h in by_hop if 0 < h < 90):
        r_truoc = max(260 if hop == 1 else r_truoc + 240,
                      r_truoc + max(240, int(18 * len(by_hop[hop]))))
        RING_R[hop] = r_truoc
    r_ngoai_cung = r_truoc
    pos = {}
    for hop, group in by_hop.items():
        radius = RING_R.get(hop, r_ngoai_cung + 100)
        if radius == 0 or len(group) == 1 and hop == 0:
            for nd in group:
                pos[nd["id"]] = (0, 0)
            continue
        offset = 0.4 * hop
        for i, nd in enumerate(sorted(group, key=lambda x: x["id"])):
            angle = offset + 2 * _math.pi * i / max(1, len(group))
            pos[nd["id"]] = (radius * _math.cos(angle), radius * _math.sin(angle))

    HOP_STYLE = {
        0: {"bg": "#f97316", "border": "#ffffff", "size": 34, "bw": 3, "fs": 14,
            "tag": "Mục tiêu đang xét"},
        1: {"bg": "#fbbf24", "border": "#f97316", "size": 22, "bw": 2, "fs": 12,
            "tag": "Cách 1 bước — có trao đổi trực tiếp"},
        2: {"bg": "#8a8175", "border": "#a8a29e", "size": 15, "bw": 1, "fs": 11,
            "tag": "Cách 2 bước, nối qua một endpoint trung gian"},
        3: {"bg": "#5f5a54", "border": "#8a8175", "size": 11, "bw": 1, "fs": 10,
            "tag": "Cách 3 bước, nối qua hai endpoint trung gian"},
    }
    ROLE_TAG = {"TARGET": "mục tiêu", "ATTACKER": "có gửi flow bị gắn cờ tấn công",
                "VICTIM": "có nhận flow bị gắn cờ tấn công", "SAFE": "không dính flow bị gắn cờ"}

    for node in nodes:
        node_id = node["id"]
        hop = int(node.get("hop", 99))
        style = HOP_STYLE.get(hop, HOP_STYLE[2])
        ip_addr = node.get("ip", f"Node {node_id}")
        role = node.get("role", "SAFE")
        x, y = pos.get(node_id, (0, 0))

        label = str(ip_addr) if hop <= 1 else " "

        net.add_node(
            node_id,
            label=label,
            x=int(x), y=int(y),
            physics=enable_physics,
            color={"background": style["bg"], "border": style["border"],
                   "highlight": {"background": "#ffffff", "border": style["bg"]}},
            size=style["size"],
            shape="dot",
            borderWidth=style["bw"],
            title=f"{ip_addr}\n{style['tag']}\nVai trò: {ROLE_TAG.get(role, role)}",
            font={"size": style["fs"], "color": "#f5f5f4", "face": "monospace"},
        )

    center_id = subgraph_data.get("center_node")
    EDGE_C = {
        "center_attack": "#c2410c",
        "center_normal": "#8a6a3f",
        "outer_attack": "#9a5a4a",
        "outer_normal": "#6b6259",
        "bo_sot": "#c8c2b8",
    }

    # Gộp luồng cùng cặp endpoint thành một cạnh.
    gop = {}
    for edge in edges:
        khoa = (edge.get("source"), edge.get("target"))
        g = gop.get(khoa)
        if g is None:
            g = gop[khoa] = {
                "source": edge.get("source"), "target": edge.get("target"),
                "src_ip": edge.get("src_ip"), "dst_ip": edge.get("dst_ip"),
                "so_luong": 0, "so_gan_co": 0,
                "so_bo_sot": 0, "so_bao_nham": 0,
                "diem_cao_nhat": None, "loai": {}, "loai_bo_sot": {},
            }
        g["so_luong"] += 1
        diem = edge.get("anomaly_score")
        if diem is not None:
            diem = float(diem)
            if g["diem_cao_nhat"] is None or diem > g["diem_cao_nhat"]:
                g["diem_cao_nhat"] = diem
        nhan_that = edge.get("nhan_that")
        if edge.get("is_anomaly"):
            g["so_gan_co"] += 1
            loai = str(edge.get("attack_type", "Unknown"))
            g["loai"][loai] = g["loai"].get(loai, 0) + 1
            if nhan_that is False:
                g["so_bao_nham"] += 1
        elif nhan_that is True:
            # Luồng tấn công bị mô hình bỏ sót.
            g["so_bo_sot"] += 1
            loai = str(edge.get("loai_that") or "tấn công")
            g["loai_bo_sot"][loai] = g["loai_bo_sot"].get(loai, 0) + 1

    # Bề dày cạnh theo số luồng.
    max_luong = max((g["so_luong"] for g in gop.values()), default=1)
    for g in gop.values():
        is_anomaly = g["so_gan_co"] > 0
        diem = g["diem_cao_nhat"]
        touches_center = g["source"] == center_id or g["target"] == center_id
        ring = "center" if touches_center else "outer"

        ty_le = _math.log1p(g["so_luong"]) / _math.log1p(max_luong)
        if is_anomaly:
            color = {"color": EDGE_C[f"{ring}_attack"], "highlight": "#ffffff"}
            day = (2.2 + 5.0 * ty_le) * (1.0 if touches_center else 0.6)
            ten_loai = max(g["loai"], key=g["loai"].get)
            dong_dau = (f"{g['so_gan_co']}/{g['so_luong']} luồng mô hình gắn "
                        f"cờ: {ten_loai}")
        elif g["so_bo_sot"]:
            # Cạnh bị bỏ sót dùng màu xám.
            color = {"color": EDGE_C["bo_sot"], "highlight": "#ffffff"}
            day = (2.0 + 4.2 * ty_le) * (1.0 if touches_center else 0.6)
            dong_dau = f"{g['so_luong']} luồng, mô hình không gắn cờ luồng nào"
        else:
            color = {"color": EDGE_C[f"{ring}_normal"], "highlight": "#ffffff"}
            day = (0.9 + 2.2 * ty_le) * (1.0 if touches_center else 0.65)
            dong_dau = f"{g['so_luong']} luồng, mô hình không gắn cờ luồng nào"

        # Nét đứt đánh dấu luồng bị bỏ sót.
        nets = {"dashes": True} if g["so_bo_sot"] else {}
        dong_lech = []
        if g["so_bo_sot"]:
            ten = max(g["loai_bo_sot"], key=g["loai_bo_sot"].get)
            dong_lech.append(f"BỎ SÓT: {g['so_bo_sot']} luồng nhãn thật là "
                             f"{ten} mà mô hình đoán lành tính")
        if g["so_bao_nham"]:
            dong_lech.append(f"BÁO NHẦM: {g['so_bao_nham']} luồng nhãn thật "
                             "lành tính mà mô hình gắn cờ")

        diem_txt = (
            f"điểm bất thường cao nhất "
            f"{decimal.Decimal(repr(float(diem))).normalize():f}"
            if diem is not None else "không có điểm bất thường")

        net.add_edge(
            g["source"], g["target"],
            color=color, width=day,
            title="\n".join([dong_dau, *dong_lech, diem_txt,
                              f"{g['src_ip']} -> {g['dst_ip']}"]),
            **nets,
        )

    return net.generate_html(notebook=False)


def plot_metrics_gauges(metrics: dict) -> go.Figure:
    # Đồng hồ đo chỉ số chính và recall tấn công.
    acc = metrics.get("accuracy")
    prec = metrics.get("precision", metrics.get("precision_macro"))
    rec = metrics.get("recall", metrics.get("recall_macro"))
    f1 = metrics.get("f1_macro", metrics.get("f1"))

    # Recall của lớp tấn công sau quy đổi nhị phân.
    attack_recall = (metrics.get("attack_recall")
                     or metrics.get("binary_equivalent", {}).get("recall"))

    ten_recall = "Recall (macro)" if attack_recall is not None else "Recall"
    gauge_data = [(name, None if value is None else float(value) * 100)
                  for name, value in (("Accuracy", acc), ("Precision", prec),
                                       (ten_recall, rec), ("Macro F1", f1),
                                       ("Recall lớp tấn công", attack_recall))
                  if not (name == "Recall lớp tấn công" and value is None)]

    n_gauge = len(gauge_data)
    fig = make_subplots(
        rows=1, cols=n_gauge,
        specs=[[{"type": "indicator"}] * n_gauge],
    )

    colors = ["#10b981", "#3b82f6", "#f59e0b", "#6366f1", "#ef4444"]

    for i, (name, value) in enumerate(gauge_data):
        if value is None:
            # Báo thiếu dữ liệu thay vì hiển thị 0%.
            fig.add_annotation(text=f"{name}<br>không có chỉ số",
                               xref="paper", yref="paper",
                               x=(i + 0.5) / n_gauge, y=0.5, showarrow=False,
                               font=dict(size=12, color=CHART["text"]))
            continue
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": name, "font": {"size": 14, "color": CHART["text"]}},
                number={"valueformat": ".3f", "suffix": "%", "font": {"size": 15, "color": CHART["text"]}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#888")),
                    bar=dict(color=colors[i]),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 50], color="rgba(255,71,87,0.15)"),
                        dict(range=[50, 80], color="rgba(255,165,2,0.15)"),
                        dict(range=[80, 100], color="rgba(0,212,170,0.15)"),
                    ],
                ),
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return _apply_axis_theme(fig)


# Màu biểu đồ theo chủ đề.
CHART = {
    "benign": "#6b8e9f", "attack": "#f97316",
    "grid": "#2a1810", "text": "#e7e5e4", "bg": "#07090e",
    "seq": [[0, "#1a0f07"], [0.25, "#3d2410"], [0.6, "#c2410c"], [1, "#fbbf24"]],
    "template": "plotly_dark",
    "cat": ["#f97316", "#fbbf24", "#dc2626", "#c2410c",
            "#6b8e9f", "#eab308", "#a3a3a3", "#7d9471",
            "#9a3412", "#d6d3d1"],
}


def _apply_axis_theme(fig):
    # Đồng bộ trục và đường lưới theo chủ đề.
    fig.update_xaxes(
        tickfont=dict(color=CHART["text"]),
        title_font=dict(color=CHART["text"]),
        gridcolor=CHART["grid"],
        linecolor=CHART["grid"],
        zerolinecolor=CHART["grid"],
    )
    fig.update_yaxes(
        tickfont=dict(color=CHART["text"]),
        title_font=dict(color=CHART["text"]),
        gridcolor=CHART["grid"],
        linecolor=CHART["grid"],
        zerolinecolor=CHART["grid"],
    )
    return fig


def apply_chart_theme(tokens: dict) -> None:
    # Cập nhật màu biểu đồ từ giao diện.
    CHART.update({
        "benign": tokens.get("chart_benign", CHART["benign"]),
        "attack": tokens.get("chart_attack", CHART["attack"]),
        "grid": tokens.get("chart_grid", CHART["grid"]),
        "text": tokens.get("chart_text", CHART["text"]),
        "bg": tokens.get("chart_bg", CHART["bg"]),
        "seq": tokens.get("chart_seq", CHART["seq"]),
        "template": tokens.get("chart_template", CHART["template"]),
        "cat": tokens.get("chart_cat", CHART["cat"]),
    })


def build_review_profile(df_result, expected_features: int = 49,
                         missing_features=None, review_budget_pct: float = 5.0) -> dict:
    # Tổng hợp đặc tính dữ liệu đầu vào.
    import numpy as np
    import pandas as pd

    out = {"notes": [], "stats": {}}
    if df_result is None or len(df_result) == 0:
        return out

    n = len(df_result)
    missing_features = list(missing_features or [])
    n_missing = len(missing_features)
    missing_ratio = n_missing / expected_features if expected_features else 0.0
    out["stats"]["rows"] = n
    out["stats"]["missing_features"] = n_missing
    out["stats"]["missing_ratio"] = missing_ratio

    # Kiểm tra mức độ đầy đủ của đặc trưng.
    if n_missing == 0:
        out["notes"].append(("ok", "Đủ đặc trưng", f"Có đầy đủ {expected_features} cột mô hình cần."))
    else:
        groups = []
        if any("IAT" in c for c in missing_features):
            groups.append("nhịp gói tin (IAT)")
        if any("TCP_FLAGS" in c for c in missing_features):
            groups.append("cờ TCP")
        if any("THROUGHPUT" in c for c in missing_features):
            groups.append("thông lượng")
        detail = f" Nhóm bị thiếu: {', '.join(groups)}." if groups else ""
        level = "warn" if missing_ratio > 0.30 else "info"
        out["notes"].append((
            level, f"Thiếu {n_missing}/{expected_features} đặc trưng",
            f"Các cột thiếu đã được điền 0.{detail} "
            + ("Tỷ lệ thiếu vượt ngưỡng nên toàn bộ dòng vào hàng đợi."
               if missing_ratio > 0.30 else
               "Tỷ lệ còn dưới ngưỡng, chưa đủ để nghi ngờ toàn bộ file."),
        ))

    # Kiểm tra nguồn định danh endpoint.
    if "endpoint_identity_source" in df_result.columns:
        synth = (df_result["endpoint_identity_source"] != "observed").mean()
        out["stats"]["synthetic_ratio"] = float(synth)
        if synth > 0.5:
            out["notes"].append((
                "warn", "Endpoint là định danh tổng hợp",
                "Bộ dữ liệu không có IP thật; endpoint được dựng từ port chỉ để chạy mô hình. "
                "Không dùng kết quả này để quy kết máy nào tấn công.",
            ))
        else:
            out["notes"].append(("ok", "Endpoint quan sát được", "Dữ liệu có địa chỉ thật trong capture."))

    # Tóm tắt biên độ bất định.
    if "uncertainty_margin" in df_result.columns:
        m = pd.to_numeric(df_result["uncertainty_margin"], errors="coerce").dropna().to_numpy()
        if len(m):
            med = float(np.median(m))
            low = float((m < 0.20).mean())
            out["stats"]["margin_median"] = med
            out["stats"]["margin_low_ratio"] = low
            if med > 0.90:
                out["notes"].append((
                    "ok", f"Mô hình chắc chắn trên phần lớn dòng (margin trung vị {med:.2f})",
                    f"Chỉ {decimal.Decimal(repr(float(low))) * 100:f}% số dòng có margin dưới 0,20. "
                    f"Hạn mức {review_budget_pct:.0f}% đang giữ lại phần phân vân nhất.",
                ))
            else:
                out["notes"].append((
                    "warn", f"Mô hình phân vân trên diện rộng (margin trung vị {med:.2f})",
                    f"{decimal.Decimal(repr(float(low))) * 100:f}% số dòng có margin dưới 0,20. Phân phối này thường gặp khi "
                    "dữ liệu khác miền huấn luyện; nên đọc kết quả dè dặt hơn.",
                ))

    # Kiểm tra sự hiện diện của nhãn thật.
    has_label = any(c in df_result.columns for c in ("Label", "label"))
    out["stats"]["has_label"] = has_label
    if not has_label:
        out["notes"].append((
            "info", "Không có nhãn thật",
            "File không kèm cột Label nên không thể biết dự đoán đúng hay sai; "
            "hàng đợi chỉ phản ánh mức thiếu chắc chắn của mô hình.",
        ))

    return out


def render_capture_timeline_html(
    df_result,
    source_label: str = "",
    status_text: str = "",
    status_ok: bool = True,
    n_windows: int = 48,
) -> str:
    # Dải mật độ tấn công dạng HTML.
    import numpy as np
    import pandas as pd

    ident = (
        '<div class="capture-id">'
        '<span class="capture-mark material-symbols-outlined">security</span>'
        '<span class="capture-name">TGN-NIDS</span>'
        '<span class="capture-desc">Temporal graph intrusion detection on NetFlow</span>'
        + (
            f'<span class="capture-status {"is-ok" if status_ok else "is-warn"}">{status_text}</span>'
            if status_text
            else ""
        )
        + "</div>"
    )

    if df_result is None or len(df_result) == 0 or "prediction" not in df_result.columns:
        return f'<div class="capture-hero">{ident}</div>'

    pred = pd.to_numeric(df_result["prediction"], errors="coerce").fillna(0).to_numpy()
    total = len(pred)
    attacks = int((pred == 1).sum())
    benign = total - attacks
    ti_le_du_doan = attacks / total if total else 0.0

    # Chia cửa sổ theo số luồng.
    bounds = np.linspace(0, total, min(n_windows, max(total, 1)) + 1).astype(int)
    bars, shares = [], []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        chunk = pred[lo:hi]
        share = float((chunk == 1).mean())
        shares.append((share, lo, hi))
    if not shares:
        return f'<div class="capture-hero">{ident}</div>'

    peak_idx = int(np.argmax([sh for sh, _, _ in shares]))
    # Chuẩn hóa chiều cao cột mật độ.
    lo_share = min(sh for sh, _, _ in shares)
    hi_share = max(sh for sh, _, _ in shares)
    span = hi_share - lo_share
    for i, (share, _, _) in enumerate(shares):
        if span > 1e-9:
            height = 8 + round((share - lo_share) / span * 88)
        else:
            height = 50
        cls = "rib-bar is-peak" if i == peak_idx else "rib-bar"
        bars.append(f'<span class="{cls}" style="height:{height}%"></span>')

    peak_share, peak_lo, peak_hi = shares[peak_idx]
    peak_n = peak_hi - peak_lo
    peak_note = (
        f"flows {peak_lo:,}&ndash;{peak_hi:,} &middot; {peak_n:,} flows &middot; "
        f"{_percent(peak_share, decimals=1, vietnamese=False)} attack"
    )
    if "prediction_label" in df_result.columns:
        window = df_result.iloc[peak_lo:peak_hi]
        atk = window[pd.to_numeric(window["prediction"], errors="coerce").fillna(0) == 1]
        if len(atk):
            top = atk["prediction_label"].astype(str).value_counts()
            if len(top):
                peak_note += f" &middot; mostly {top.index[0]}"

    def _short(v: int) -> str:
        return f"{v/1000:.1f}k".replace(".0k", "k") if v >= 10000 else f"{v:,}"

    n_ticks = 4
    ticks = "".join(
        f"<span>{_short(int(i * total / n_ticks))}</span>" for i in range(n_ticks + 1)
    )

    # Tính mốc trivial-F1 khi có nhãn thật.
    cot_nhan = next((c for c in df_result.columns
                     if str(c).strip().lower() == "label"), None)
    if cot_nhan is not None:
        that = pd.to_numeric(df_result[cot_nhan], errors="coerce")
        ti_le = float((that == 1).mean())
        nguon_moc = "nhãn thật"
    else:
        ti_le = None
        nguon_moc = None
    moc_tam_thuong = (2.0 * ti_le / (1.0 + ti_le)) if ti_le else None
    nhan_moc = ("trivial-F1 baseline" if nguon_moc
                else "trivial-F1 baseline (cần nhãn)")

    return f"""<div class="capture-hero">
  {ident}
  <div class="capture-meta">capture window &middot; {total:,} flows{(' &middot; ' + source_label) if source_label else ''}
    <span class="capture-scale">attack share per window &middot; scale {_percent(lo_share, decimals=1, vietnamese=False)}&ndash;{_percent(hi_share, decimals=1, vietnamese=False)}</span>
  </div>
  <div class="capture-ribbon">{''.join(bars)}</div>
  <div class="capture-axis">{ticks}</div>
  <div class="capture-axis-note">flow order &middot; oldest to newest</div>
  <div class="capture-stats">
    <div class="stat"><span class="stat-num">{benign:,}</span><span class="stat-lab">normal</span></div>
    <div class="stat"><span class="stat-num is-threat">{attacks:,}</span><span class="stat-lab">threat</span></div>
    <div class="stat"><span class="stat-num is-threat">{_percent(ti_le_du_doan, vietnamese=False)}</span><span class="stat-lab">anomaly ratio</span></div>
    <div class="stat"><span class="stat-num">{_percent(moc_tam_thuong, vietnamese=False)}</span><span class="stat-lab">{nhan_moc}</span></div>
  </div>
  <div class="capture-peak">highest attack share: {peak_note}</div>
</div>"""


def render_alert_table_html(alerts_df: pd.DataFrame, max_rows: int = 20) -> str:
    # Tạo bảng cảnh báo HTML cho phân tích theo lô.
    head = (
        '<div class="alert-feed">'
        '<div class="alert-feed-head">'
        '<span class="material-symbols-outlined">crisis_alert</span>'
        "<span>Alert feed</span>"
        f'<span class="alert-feed-count">{len(alerts_df):,} flows flagged</span>'
        "</div>"
    )
    if alerts_df is None or len(alerts_df) == 0:
        return head + '<div class="alert-feed-empty">No attacks detected in current data.</div></div>'

    frame = alerts_df.head(max_rows)
    cols = list(frame.columns)
    endpoint_cols = {cols[0], cols[1]} if len(cols) >= 2 else set()

    COL_HELP = {
        "Port": "Cổng đích của flow. Gợi ý dịch vụ bị nhắm tới (80 HTTP, 443 HTTPS, 22 SSH).",
        "Attack Type": "Lớp tấn công mô hình dự đoán, trong 10 lớp của NF-UNSW-NB15. "
                       "Đây là dự đoán, không phải nhãn thật.",
        "Confidence": "Xác suất mô hình gán cho lớp nó chọn. Cao không đồng nghĩa với đúng: "
                       "dự đoán bỏ sót vẫn có thể có confidence rất cao.",
        "Anomaly Score": "Điểm bất thường do bộ giải mã cạnh sinh ra, càng cao càng lệch khỏi "
                         "hành vi bình thường đã học. Thang điểm riêng của từng lần chạy, "
                         "không so trực tiếp giữa hai bộ dữ liệu.",
    }
    ENDPOINT_HELP = ("Đầu kia của flow. Nếu ghi 'port NNN' thì đây là định danh ẩn danh dựng "
                     "từ cổng, không phải địa chỉ máy thật.")

    ths = []
    for c in cols:
        cls = "c-endpoint" if c in endpoint_cols else "c-attack"
        if c == "Port":
            cls = "c-batch"
        elif c in ("Confidence", "Anomaly Score"):
            cls = "c-conf"
        tip = ENDPOINT_HELP if c in endpoint_cols else COL_HELP.get(c, "")
        ths.append(f'<th class="{cls}" title="{_esc(tip)}">{_esc(c)}</th>')

    rows = []
    for _, record in frame.iterrows():
        tds = []
        for c in cols:
            raw = record[c]
            if c in endpoint_cols:
                shown, is_anon = _format_endpoint(raw)
                anon = " is-anon" if is_anon else ""
                tds.append(f'<td class="c-endpoint{anon}" title="{_esc(raw)}">{_esc(shown)}</td>')
            elif c == "Port":
                tds.append(f'<td class="c-batch">{_esc(raw)}</td>')
            elif c in ("Confidence", "Anomaly Score"):
                tds.append(f'<td class="c-conf">{_esc(raw)}</td>')
            else:
                tds.append(f'<td class="c-attack">{_esc(raw)}</td>')
        rows.append("<tr class=\"alert-row\">" + "".join(tds) + "</tr>")

    return (
        head
        + '<div class="alert-feed-body"><table class="alert-table">'
        + "<thead><tr>" + "".join(ths) + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )


def plot_realtime_timeline(accumulated_df: pd.DataFrame, window_size: int = 10, max_windows: int = 40) -> go.Figure:
    # Biểu đồ luồng theo cửa sổ trượt.
    df = accumulated_df.copy()

    # Ưu tiên nhãn thật, nếu có.
    label_col = "Label" if "Label" in df.columns else "prediction"
    if label_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            template=CHART["template"],
            font=dict(color=CHART["text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        return fig

    df["time_idx"] = range(len(df))
    df["window"] = df["time_idx"] // max(1, window_size)
    df["type"] = df[label_col].map({0: "Benign", 1: "Attack"})

    grouped = df.groupby(["window", "type"]).size().reset_index(name="count")

    # Chỉ giữ các cửa sổ gần nhất.
    max_w = grouped["window"].max() if len(grouped) > 0 else 0
    min_w = max(0, max_w - max_windows + 1)
    grouped = grouped[grouped["window"] >= min_w]

    fig = go.Figure()

    # Màu cho từng nhóm lưu lượng.
    colors = {"Benign": CHART["benign"], "Attack": CHART["attack"]}

    for t in ["Benign", "Attack"]:
        mask = grouped["type"] == t
        fig.add_trace(go.Scatter(
            x=grouped[mask]["window"],
            y=grouped[mask]["count"],
            name=t,
            mode="lines",
            fill="tonexty" if t == "Attack" else "tozeroy",
            line=dict(color=colors.get(t, "#888"), width=2),
            fillcolor=CHART["benign"] if t == "Benign" else CHART["attack"], opacity=0.30,
        ))

    fig.update_layout(
        title=dict(
            text="live traffic · streaming",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        xaxis_title="Batch",
        yaxis_title="Flows",
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=CHART["text"], family="IBM Plex Sans, sans-serif"),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
        xaxis=dict(
            range=[min_w - 0.5, max_w + 0.5] if max_w > 0 else None,
        ),
    )

    return _apply_axis_theme(fig)


def plot_realtime_attack_rate(batch_history: list) -> go.Figure:
    # Biểu đồ tỷ lệ tấn công theo lô.
    if not batch_history:
        fig = go.Figure()
        fig.update_layout(
            template=CHART["template"],
            font=dict(color=CHART["text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
        )
        return fig

    batch_indices = [b["batch_idx"] for b in batch_history]
    attack_rates = [b["attack_rate"] for b in batch_history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=batch_indices,
        y=attack_rates,
        mode="lines+markers",
        name="Attack Rate",
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=5, color="#ef4444"),
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.1)",
    ))

    # Đường trung bình qua các lô.
    if len(attack_rates) > 1:
        avg_rate = sum(attack_rates) / len(attack_rates)
        fig.add_hline(
            y=avg_rate,
            line_dash="dash",
            line_color="rgba(249,115,22,0.6)",
            annotation_text=f"Avg: {decimal.Decimal(repr(float(avg_rate))).normalize():f}%",
            annotation_font=dict(color="#f59e0b", size=11),
        )

    fig.update_layout(
        title=dict(
            text="attack rate per batch",
            font=dict(size=13, color=CHART["text"], family="IBM Plex Mono, monospace"),
            x=0, xanchor="left",
        ),
        xaxis_title="Batch #",
        yaxis_title="Attack Rate (%)",
        yaxis=dict(range=[0, max(100, max(attack_rates) * 1.2) if attack_rates else 100]),
        template=CHART["template"],
        # Đồng bộ chữ với chủ đề.
        font=dict(color=CHART["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
        showlegend=False,
    )

    return _apply_axis_theme(fig)


def _format_endpoint(value: str) -> tuple:
    # Rút gọn endpoint và đánh dấu định danh tổng hợp.
    text = str(value)
    if text.startswith("ANONYMOUS_"):
        tail = text.rsplit("_", 1)[-1]
        if "_PORT_" in text:
            return f"port {tail}", True
        return f"flow {tail}", True
    return text, False


def render_live_alert_feed_html(recent_alerts: list, max_display: int = 15) -> str:
    # Tạo bảng cảnh báo HTML cho luồng thời gian thực.
    head = (
        '<div class="alert-feed">'
        '<div class="alert-feed-head">'
        '<span class="material-symbols-outlined">notifications_active</span>'
        "<span>Live alert feed</span>"
        f'<span class="alert-feed-count">{len(recent_alerts):,} total</span>'
        "</div>"
    )

    if not recent_alerts:
        return (
            head
            + '<div class="alert-feed-empty">No alerts yet. Alerts appear here as '
            "batches are scored.</div></div>"
        )

    display_alerts = recent_alerts[-max_display:][::-1]

    rows = []
    for i, alert in enumerate(display_alerts):
        src, src_anon = _format_endpoint(alert.get("src_ip", "—"))
        dst, dst_anon = _format_endpoint(alert.get("dst_ip", "—"))
        attack = str(alert.get("attack_type", "Unknown"))
        conf = alert.get("confidence")
        batch = str(alert.get("batch", "—"))
        newest = " is-newest" if i == 0 else ""
        src_cls = " is-anon" if src_anon else ""
        dst_cls = " is-anon" if dst_anon else ""
        rows.append(
            f'<tr class="alert-row{newest}">'
            f'<td class="c-batch">{_esc(batch)}</td>'
            f'<td class="c-endpoint{src_cls}" title="{_esc(alert.get("src_ip", ""))}">{_esc(src)}</td>'
            f'<td class="c-endpoint{dst_cls}" title="{_esc(alert.get("dst_ip", ""))}">{_esc(dst)}</td>'
            f'<td class="c-attack">{_esc(attack)}</td>'
            f'<td class="c-conf">{"—" if conf is None else f"{decimal.Decimal(repr(float(conf))) * 100:f}%"}</td>'
            "</tr>"
        )

    return (
        head
        + '<div class="alert-feed-body"><table class="alert-table">'
        "<thead><tr>"
        '<th class="c-batch" title="Số thứ tự lô flow đã xử lý, tăng dần theo nhịp stream.">Batch</th>'
        '<th class="c-endpoint" title="Bên khởi tạo flow. Ghi \'port NNN\' nghĩa là định danh ẩn danh dựng từ cổng, không phải máy thật.">Source</th>'
        '<th class="c-endpoint" title="Bên nhận flow. Cùng quy ước với cột Source.">Destination</th>'
        '<th class="c-attack" title="Lớp tấn công mô hình dự đoán. Là dự đoán, không phải nhãn thật.">Attack</th>'
        '<th class="c-conf" title="Xác suất mô hình gán cho lớp nó chọn. Cao không đồng nghĩa với đúng.">Conf.</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )


def create_realtime_metrics_html(
    total_flows: int,
    benign: int,
    attacks: int,
    attack_rate: float,
    is_live: bool = False,
) -> str:
    # Tạo thẻ chỉ số HTML cho thời gian thực.
    live_class = "pulse" if is_live else ""

    # Chọn màu theo tỷ lệ tấn công.
    if attack_rate > 20:
        rate_color = "#dc2626"
    elif attack_rate > 5:
        rate_color = "#fbbf24"
    else:
        rate_color = "#6b8e9f"

    rate_text = f"{attack_rate:.1f}%"
    total_str = f"{total_flows:,}"
    benign_str = f"{benign:,}"
    attacks_str = f"{attacks:,}"

    cards = [
        '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px;">',
        f'    <div class="metric-card {live_class}">',
        f'        <p class="metric-value">{total_str}</p>',
        '        <p class="metric-label">Total Flows</p>',
        '    </div>',
        '    <div class="metric-card">',
        f'        <p class="metric-value" style="color: #6b8e9f;">{benign_str}</p>',
        '        <p class="metric-label">Benign</p>',
        '    </div>',
        f'    <div class="metric-card {live_class}">',
        f'        <p class="metric-value" style="color: #ef4444;">{attacks_str}</p>',
        '        <p class="metric-label">Attacks Detected</p>',
        '    </div>',
        '    <div class="metric-card">',
        f'        <p class="metric-value" style="color: {rate_color};">{rate_text}</p>',
        '        <p class="metric-label">Attack Rate</p>',
        '    </div>',
        '</div>',
    ]
    return "\n".join(cards)


def render_df_html(frame, title: str = "", icon: str = "table_rows",
                   max_rows: int = 50, note: str = "") -> str:
    # Tạo bảng HTML theo phong cách giao diện.
    if frame is None or len(frame) == 0:
        return (f'<div class="alert-feed"><div class="alert-feed-head">'
                f'<span class="material-symbols-outlined">{_esc(icon)}</span><span>{_esc(title)}</span></div>'
                f'<div class="alert-feed-empty">Không có dữ liệu.</div></div>')

    shown = frame.head(max_rows)
    cols = [str(c) for c in shown.columns]

    def fmt(v):
        # Rút gọn số hiển thị trong bảng.
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
        return str(v)

    ths = "".join(f'<th class="c-attack">{_esc(c)}</th>' for c in cols)
    rows = []
    for _, rec in shown.iterrows():
        tds = "".join(f'<td class="c-attack">{_esc(fmt(rec[c]))}</td>' for c in shown.columns)
        rows.append(f'<tr class="alert-row">{tds}</tr>')

    count = (f'<span class="alert-feed-count">{len(shown):,}/{len(frame):,} dòng</span>'
             if len(frame) > len(shown) else
             f'<span class="alert-feed-count">{len(frame):,} dòng</span>')
    head = (f'<div class="alert-feed-head">'
            f'<span class="material-symbols-outlined">{_esc(icon)}</span>'
            f'<span>{_esc(title)}</span>{count}</div>')
    foot = f'<div class="alert-feed-note">{_esc(note)}</div>' if note else ""
    return (f'<div class="alert-feed df-table">{head}'
            f'<div class="alert-feed-body"><table class="alert-table">'
            f"<thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{foot}</div>")
