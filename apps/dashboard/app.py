import html as _html
import streamlit as st
import pandas as pd
import os
import sys
import time


def _esc(value) -> str:
    # An toàn khi chèn giá trị vào HTML.
    return _html.escape("" if value is None else str(value), quote=True)


# Thêm mã nguồn cục bộ vào đường dẫn import.
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(DASHBOARD_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
APPS_DIR = os.path.join(PROJECT_DIR, "apps")
for p in [SRC_DIR, APPS_DIR, DASHBOARD_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from inference_engine import NIDSInferenceEngine, load_saved_metrics, get_all_metrics_for_comparison, MODEL_CONFIGS, MISSING_FEATURE_RATIO_THRESHOLD, format_exact_percent, format_exact_number
    from visualization import (
        plot_traffic_timeline,
        plot_attack_distribution,
        plot_confusion_matrix,
        plot_metrics_comparison,
        build_metrics_comparison_legend_html,
        plot_feature_importance,
        render_ego_network_html,
        loc_canh_theo_nguong,
        plot_metrics_gauges,
        render_alert_table_html,
        render_capture_timeline_html,
        _format_endpoint,
        render_df_html,
        build_review_profile,
        apply_chart_theme,
        plot_realtime_timeline,
        plot_realtime_attack_rate,
        render_live_alert_feed_html,
        create_realtime_metrics_html,
    )
except ImportError:
    from dashboard.inference_engine import NIDSInferenceEngine, load_saved_metrics, get_all_metrics_for_comparison, MODEL_CONFIGS, MISSING_FEATURE_RATIO_THRESHOLD, format_exact_percent, format_exact_number
    from dashboard.visualization import (
        plot_traffic_timeline,
        plot_attack_distribution,
        plot_confusion_matrix,
        plot_metrics_comparison,
        build_metrics_comparison_legend_html,
        plot_feature_importance,
        render_ego_network_html,
        loc_canh_theo_nguong,
        plot_metrics_gauges,
        render_alert_table_html,
        render_capture_timeline_html,
        _format_endpoint,
        render_df_html,
        build_review_profile,
        apply_chart_theme,
        plot_realtime_timeline,
        plot_realtime_attack_rate,
        render_live_alert_feed_html,
        create_realtime_metrics_html,
    )

SAMPLES_DIR = os.path.join(PROJECT_DIR, "data", "samples")

# Cấu hình trang Streamlit.
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="security",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Nạp CSS của dashboard.
css_path = os.path.join(DASHBOARD_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f_css:
        st.markdown(f"<style>{f_css.read()}</style>", unsafe_allow_html=True)

# Đồng bộ giao diện và biểu đồ theo chủ đề.
try:
    from theme import THEME_TOKENS, build_theme_css, build_dataframe_css
except ImportError:
    from dashboard.theme import THEME_TOKENS, build_theme_css, build_dataframe_css

_theme_choice = "Sáng" if st.session_state.get("ui_light", False) else "Tối"
THEME = THEME_TOKENS[_theme_choice]
st.markdown(build_theme_css(THEME), unsafe_allow_html=True)
st.markdown(build_dataframe_css(THEME), unsafe_allow_html=True)
apply_chart_theme(THEME)


# Thanh điều khiển bên.
with st.sidebar:
    st.markdown("""
    <div style="background: var(--bg-card); border: 1.5px solid var(--border-accent); border-radius: 14px; padding: 18px 12px; text-align: center; margin-bottom: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <div style="font-size: 1.12rem; font-weight: 900; color: var(--text-highlight); letter-spacing: -0.01em; font-family: 'IBM Plex Sans', sans-serif; white-space: nowrap;">
            <span class="material-symbols-outlined panel-mark">security</span> NIDS CONTROL PANEL
        </div>
        <div style="font-size: 0.72rem; color: var(--text-main); font-family: 'IBM Plex Mono', monospace; margin-top: 5px; letter-spacing: 0.05em;">
            MODEL &middot; DATA SOURCE &middot; XAI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chọn phiên bản mô hình.
    from dashboard.inference_engine import MODEL_CONFIGS
    model_version = st.selectbox(
        ":material/psychology: Model Architecture",
        options=list(MODEL_CONFIGS.keys()),
        help="Pick a TGN model fully trained on NF-UNSW-NB15.",
    )

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Chọn chế độ phân tích và chủ đề.
    _om_label, _om_toggle = st.columns([5, 2])
    with _om_toggle:
        st.toggle(
            ":material/dark_mode:" if st.session_state.get("ui_light", False) else ":material/light_mode:",
            key="ui_light",
            help="Switch between dark and light theme.",
        )

    analysis_mode = st.radio(
        ":material/bolt: Operation Mode",
        options=["Batch Analysis", "Realtime Streaming"],
        help="Batch: analyse a whole log file. Realtime: replay the traffic as a live stream.",
    )

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Giá trị mặc định cho nguồn dữ liệu.
    uploaded_file = None
    rt_data_file = None
    rt_data_label = None
    rt_source = "Built-in Scenario"
    rt_batch_size = 15
    rt_speed = 1.0

    # Các kịch bản mẫu đi kèm dashboard.
    sample_files_map = {
        "1. In-distribution - NF-UNSW-NB15-v3 test split (10,000 flows)": "demo_1_cung_phan_phoi.csv",
        "2. Unseen dataset - NF-CSE-CIC-IDS2018-v3, with addresses (10,000 flows)": "demo_2_bo_du_lieu_la.csv",
        "3. Address columns missing - NF-CSE-CIC-IDS2018-v2 (10,000 flows)": "demo_3_thieu_cot_dia_chi.csv",
    }

    if analysis_mode == "Batch Analysis":
        data_source = st.radio(
            ":material/database: Data Source",
            options=["Built-in Demo Scenario", "Upload CSV / Parquet File"],
            help="Choose the data source to analyse.",
        )

        if data_source == "Upload CSV / Parquet File":
            with st.expander("Upload file format", expanded=False):
                st.markdown(
                    "The model was trained on NetFlow data following the nProbe/Zeek schema "
                    "(schema <b>NF-</b>, for example "
                    "<code style='background:transparent;color:var(--neon-cyan);padding:0;'>IPV4_SRC_ADDR</code>, "
                    "<code style='background:transparent;color:var(--neon-cyan);padding:0;'>L4_SRC_PORT</code>, "
                    "<code style='background:transparent;color:var(--neon-cyan);padding:0;'>IN_BYTES</code>...). "
                    "When you upload a file there are two cases:<br><br>"
                    "1. File matches the <b>NF-</b> schema, inference runs in full and accuracy matches the published figures.<br>"
                    "2. Any other format (CICFlowMeter/CICIDS2017, raw Zeek conn.log, Argus, custom "
                    "logs) is "
                    "<span style='color:#f87171;font-weight:700;'>NOT recognised</span>, cả 49 "
                    "feature columns are filled with 0, the model still runs but the result is "
                    "<span style='color:#f87171;font-weight:700;'>NO longer meaningful</span>."
                    "<br><br>"
                    "After uploading, the badge above the metrics table reports how many columns are missing.",
                    unsafe_allow_html=True,
                )
            uploaded_file = st.file_uploader(
                "Upload NetFlow CSV / Parquet",
                type=["csv", "parquet", "tsv", "txt"],
                help="NetFlow data file (.csv or .parquet from nProbe, Wireshark, Zeek).",
            )
            selected_scenario = "Uploaded File"
            selected_scenario_file = None
        else:
            uploaded_file = None
            selected_scenario = st.selectbox(
                "Telemetry Scenario",
                options=list(sample_files_map.keys()),
                help="Các kịch bản được cắt trực tiếp từ tập Test NF-UNSW-NB15-v3 hoặc dữ liệu ToN-IoT/CIC-IDS2018 ngoài.",
            )
            selected_scenario_file = sample_files_map[selected_scenario]
    else:
        data_source = "Realtime"
        uploaded_file = None

        # Chọn nguồn phát luồng.
        st.markdown("<div style='color: #f43f5e; font-weight: 800; font-size: 0.95rem; margin-bottom: 6px;'><span class='material-symbols-outlined'>sensors</span> REALTIME STREAM CONFIG</div>", unsafe_allow_html=True)

        rt_source = st.radio(
            "Stream Source",
            options=["Built-in Scenario", "Upload Stream File"],
            horizontal=True,
            help="Choose the source of the live stream.",
        )

        if rt_source == "Upload Stream File":
            rt_uploaded = st.file_uploader(
                "Upload Custom Stream CSV / Parquet",
                type=["csv", "parquet"],
                key="rt_upload_widget",
                help="NetFlow data file to replay as a live stream.",
            )
            rt_data_file = None
        else:
            rt_uploaded = None
            rt_data_label = st.selectbox(
                "Data Stream Scenario",
                options=list(sample_files_map.keys()),
                help="Choose a traffic scenario to stream.",
            )
            rt_data_file = sample_files_map[rt_data_label]

        rt_batch_size = st.slider(
            "Batch Size (Flows / Tick)",
            min_value=5, max_value=50, value=15, step=5,
            help="Number of flows handled per tick.",
        )

        rt_speed = st.slider(
            "Stream Interval (sec)",
            min_value=0.5, max_value=5.0, value=1.0, step=0.5,
            help="Delay between batches, in seconds.",
        )

        # Điều khiển phát luồng.
        col_start, col_pause, col_stop = st.columns(3)

        with col_start:
            if st.button("Start", width="stretch", type="primary"):
                st.session_state["rt_active"] = True
                st.session_state["rt_paused"] = False
                if st.session_state.get("rt_complete", False):
                    st.session_state["rt_index"] = 0
                    st.session_state["rt_accumulated"] = pd.DataFrame()
                    st.session_state["rt_alerts"] = []
                    st.session_state["rt_batch_history"] = []
                    st.session_state["rt_tick"] = 0
                    st.session_state["rt_complete"] = False
                    st.session_state["force_memory_reset"] = True

        with col_pause:
            if st.button("Pause", width="stretch"):
                st.session_state["rt_paused"] = True

        with col_stop:
            if st.button("Stop", width="stretch"):
                st.session_state["rt_active"] = False
                st.session_state["rt_paused"] = False
                st.session_state["rt_index"] = 0
                st.session_state["rt_accumulated"] = pd.DataFrame()
                st.session_state["rt_alerts"] = []
                st.session_state["rt_batch_history"] = []
                st.session_state["rt_tick"] = 0
                st.session_state["rt_complete"] = False
                st.session_state["force_memory_reset"] = True

        # Tiến độ phát luồng.
        if st.session_state.get("rt_active", False):
            if st.session_state.get("rt_paused", False):
                st.markdown("<span style='color:var(--neon-amber);font-weight:700;'>PAUSED</span>", unsafe_allow_html=True)
            else:
                total_flows_in_file = st.session_state.get("rt_total_flows", 0)
                current_idx = st.session_state.get("rt_index", 0)
                if total_flows_in_file > 0:
                    pct = min(100, current_idx / total_flows_in_file * 100)
                    st.markdown(f"""
                    <div style="color:var(--text-main);font-weight:700;font-family:monospace;margin-top:10px;">
                        <span class="live-indicator"></span> LIVE STREAMING ACTIVE
                    </div>
                    <div style="color:var(--text-main);font-size:0.85rem;margin:6px 0;">
                        Processed: <strong>{current_idx:,}</strong> / {total_flows_in_file:,} flows ({pct:.1f}%)
                    </div>
                    <div class="stream-progress">
                        <div class="stream-progress-bar" style="width:{pct:.1f}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)
        elif st.session_state.get("rt_complete", False):
            st.markdown("<span style='color:var(--text-main);font-weight:700;'><span class='material-symbols-outlined'>task_alt</span> STREAM COMPLETED</span>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Chọn trạng thái bộ nhớ trước khi xử lý dữ liệu mới.
    st.markdown("<div style='color: var(--neon-cyan); font-weight: 800; font-size: 0.95rem; margin-bottom: 6px;'><span class='material-symbols-outlined'>memory</span> MEMORY STATE</div>", unsafe_allow_html=True)
    memory_mode = st.radio(
        "Per-node memory initialisation",
        options=["Restore from checkpoint", "Wipe to zero"],
        key="memory_mode",
        help=(
            "Restore: reload the memory table saved in the checkpoint, so the model "
            "starts with interaction history already in place. "
            "Wipe: set every memory vector to 0, simulating a first deployment "
            "on a network never observed before."
        ),
    )
    st.caption(
        "Interaction history is already present, so results look better than a real deployment."
        if memory_mode == "Restore from checkpoint"
        else "No history at all, close to deploying on a brand new network."
    )

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Cấu hình vùng lân cận cho XAI.
    st.markdown("<div style='color: var(--neon-cyan); font-weight: 800; font-size: 0.95rem; margin-bottom: 6px;'><span class='material-symbols-outlined'>hub</span> NEIGHBOR ANALYSIS</div>", unsafe_allow_html=True)
    k_hops = st.slider(
        "Neighbourhood width",
        min_value=1, max_value=3, value=1,
        help=(
            "When an endpoint is analysed, the system draws the endpoints related to it. "
            "This value decides how many connection hops to reach out."
        ),
    )
    _khop_desc = {
        1: "1 hop, only endpoints that talk directly to the target",
        2: "2 hops, adds the endpoints linked to the first ring",
        3: "3 hops, adds one more ring spreading out from the second",
    }
    st.caption(_khop_desc[k_hops])

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='color: var(--neon-cyan); font-weight: 800; font-size: 0.95rem; margin-bottom: 6px;'><span class='material-symbols-outlined'>rule</span> REVIEW QUEUE</div>", unsafe_allow_html=True)
    review_budget = st.slider(
        "Review capacity (% of flows)",
        min_value=1, max_value=25, value=5, step=1,
        help=(
            "Maximum rows put in the queue, as a percentage of all flows. "
            "The system ranks by uncertainty and keeps only the top slice, "
            "so the workload stays under control."
        ),
    )
    st.caption(f"Keeping the {review_budget}% least certain rows; the rest stay out of the queue.")

    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, var(--border-accent), transparent); margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Tóm tắt chỉ số của mô hình đang chọn.
    metrics = load_saved_metrics(model_version)
    if metrics:
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1_macro", metrics.get("f1"))
        lat = metrics.get("latency_ms_per_100_flows")
        st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-accent); border-radius: 12px; padding: 14px; margin-top: 6px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-bottom: 8px;">
                <span class="material-symbols-outlined">bolt</span> MODEL SPECS SUMMARY
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: var(--text-main); font-size: 0.85rem;">Accuracy:</span>
                <span style="color: var(--neon-cyan); font-weight: 800; font-family: monospace; font-size: 0.95rem;">{format_exact_percent(acc)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: var(--text-main); font-size: 0.85rem;">Macro F1:</span>
                <span style="color: var(--text-main); font-weight: 800; font-family: monospace; font-size: 0.95rem;">{format_exact_percent(f1)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: var(--text-main); font-size: 0.85rem;">Latency:</span>
                <span style="color: var(--neon-amber); font-weight: 800; font-family: monospace; font-size: 0.95rem;">{format_exact_number(lat, " ms")}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _fmt_score(value) -> str:
    # Định dạng điểm bất thường để hiển thị.
    if value is None:
        return "chưa có điểm"
    try:
        return f"điểm {format_exact_number(value)}"
    except (TypeError, ValueError):
        return "chưa có điểm"


# Tải dữ liệu và bộ suy luận có cache.
@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data(filename: str = "demo_1_cung_phan_phoi.csv"):
    # Đọc một kịch bản mẫu.
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    st.error(f"Không tìm thấy tệp dữ liệu mẫu: {filename}")
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_streaming_data(filename: str):
    # Đọc dữ liệu dùng để phát luồng.
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    else:
        st.error(f"Không tìm thấy file: {filename}")
        return pd.DataFrame()


@st.cache_resource
def get_engine(version: str):
    # Dùng chung bộ suy luận trong phiên.
    return NIDSInferenceEngine(version=version, device="cpu")


# Khởi tạo trạng thái phát luồng.
for key, default in [
    ("rt_active", False),
    ("rt_paused", False),
    ("rt_index", 0),
    ("rt_accumulated", pd.DataFrame()),
    ("rt_alerts", []),
    ("rt_batch_history", []),
    ("rt_tick", 0),
    ("rt_complete", False),
    ("rt_total_flows", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Làm mới trạng thái khi đổi mô hình hoặc dữ liệu.
if analysis_mode == "Batch Analysis":
    selected_source_identity = (
        f"upload:{uploaded_file.name}:{uploaded_file.size}"
        if uploaded_file is not None
        else f"scenario:{selected_scenario_file}"
    )
else:
    selected_source_identity = (
        f"rt_upload:{rt_uploaded.name}:{rt_uploaded.size}"
        if rt_source == "Upload Stream File" and rt_uploaded is not None
        else f"rt_scenario:{rt_data_file}"
    )
analysis_context = (model_version, analysis_mode, selected_source_identity)
if st.session_state.get("analysis_context") != analysis_context:
    st.session_state["analysis_context"] = analysis_context
    st.session_state.pop("xai_results", None)
    st.session_state["force_memory_reset"] = True
    st.session_state["rt_active"] = False
    st.session_state["rt_paused"] = False
    st.session_state["rt_index"] = 0
    st.session_state["rt_accumulated"] = pd.DataFrame()
    st.session_state["rt_alerts"] = []
    st.session_state["rt_batch_history"] = []
    st.session_state["rt_tick"] = 0
    st.session_state["rt_complete"] = False
    st.session_state["rt_total_flows"] = 0


# Nạp dữ liệu cho lần suy luận hiện tại.
engine = get_engine(model_version)


def dat_lai_bo_nho(bo_may, che_do: str) -> None:
    """Khôi phục hoặc xóa bộ nhớ TGN và bộ đếm thời gian."""
    if che_do == "Wipe to zero":
        if getattr(bo_may, "is_model_loaded", False) and hasattr(bo_may, "memory_module"):
            bo_may.memory_module.reset_state()
        bo_may._t_counter = 0.0
    else:
        bo_may.reset_memory()

df_raw = pd.DataFrame()
df_result = pd.DataFrame()
source_display_name = ""

if analysis_mode == "Batch Analysis":
    if data_source == "Upload CSV / Parquet File" and uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                df_raw = pd.read_parquet(uploaded_file)
            else:
                try:
                    df_raw = pd.read_csv(uploaded_file)
                except Exception:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")
            df_raw.columns = df_raw.columns.str.strip()
            source_display_name = f"Uploaded File: {uploaded_file.name}"
            st.sidebar.success(f"Đã tải: {uploaded_file.name} ({len(df_raw):,} flows)")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")
            df_raw = load_dashboard_data()
            source_display_name = "Fallback Demo Data"
    elif data_source == "Upload CSV / Parquet File":
        # Hiển thị mẫu mặc định khi chưa có tệp tải lên.
        st.info("Chưa chọn tệp tải lên. Đang hiển thị mẫu NF-UNSW-NB15-v3, "
                "cùng phân phối với dữ liệu mô hình đã học. "
                "Kéo thả tệp CSV hoặc Parquet vào khung Upload ở thanh bên trái để phân tích dữ liệu của bạn.")
        df_raw = load_dashboard_data("demo_1_cung_phan_phoi.csv")
        source_display_name = "NF-UNSW-NB15-v3 (mẫu mặc định khi chưa tải tệp)"
    else:
        df_raw = load_dashboard_data(selected_scenario_file)
        source_display_name = selected_scenario

    # Đặt lại bộ nhớ khi đổi tệp.
    if uploaded_file is not None:
        batch_file_identity = f"upload:{uploaded_file.name}:{uploaded_file.size}"
    else:
        batch_file_identity = f"scenario:{source_display_name}"
    if (
        st.session_state.get("last_batch_file_identity") != batch_file_identity
        or st.session_state.get("force_memory_reset", False)
    ):
        dat_lai_bo_nho(engine, memory_mode)
        st.session_state["last_batch_file_identity"] = batch_file_identity
        st.session_state["force_memory_reset"] = False

    # Chấm từng lô với bộ nhớ đã được đặt lại.
    with st.spinner("Running TGN inference..."):
        dat_lai_bo_nho(engine, memory_mode)
        engine.review_budget_pct = float(review_budget)
        df_result = engine.predict(df_raw)
else:
    # Chuẩn bị dữ liệu để phát luồng.
    if rt_source == "Upload Stream File" and rt_uploaded is not None:
        try:
            if rt_uploaded.name.endswith(".parquet"):
                rt_full_df = pd.read_parquet(rt_uploaded)
            else:
                rt_full_df = pd.read_csv(rt_uploaded)
            rt_full_df.columns = rt_full_df.columns.str.strip()
            st.session_state["rt_total_flows"] = len(rt_full_df)
            source_display_name = f"Stream: {rt_uploaded.name}"
            st.sidebar.success(f"Đã nạp stream: {rt_uploaded.name} ({len(rt_full_df):,} flows)")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file stream: {e}")
            rt_full_df = pd.DataFrame()
            source_display_name = "Stream Error"
    elif rt_data_file:
        rt_full_df = load_streaming_data(rt_data_file)
        st.session_state["rt_total_flows"] = len(rt_full_df)
        source_display_name = rt_data_label
    else:
        rt_full_df = pd.DataFrame()
        source_display_name = "Realtime Stream"

    # Đặt lại bộ nhớ khi đổi nguồn luồng.
    if rt_source == "Upload Stream File" and rt_uploaded is not None:
        rt_file_identity = f"rt_upload:{rt_uploaded.name}:{rt_uploaded.size}"
    elif rt_data_file:
        rt_file_identity = f"rt_scenario:{rt_data_file}"
    else:
        rt_file_identity = "rt_none"
    if (
        st.session_state.get("last_rt_file_identity") != rt_file_identity
        or st.session_state.get("force_memory_reset", False)
    ):
        dat_lai_bo_nho(engine, memory_mode)
        st.session_state["last_rt_file_identity"] = rt_file_identity
        st.session_state["force_memory_reset"] = False


# Cảnh báo khi endpoint là định danh tổng hợp.
_notice_df = (
    df_result
    if analysis_mode == "Batch Analysis"
    else st.session_state.get("rt_accumulated", pd.DataFrame())
)
if (
    len(_notice_df) > 0
    and "endpoint_identity_source" in _notice_df.columns
    and (_notice_df["endpoint_identity_source"] != "observed").any()
):
    st.warning(
        "Tệp này không có địa chỉ máy thật. Dashboard đang dùng endpoint tổng hợp "
        "để kiểm tra luồng xử lý và giao diện; các KPI bên dưới không phải kết quả "
        "đánh giá hiệu năng khoa học của mô hình đồ thị."
    )

# Cho biết nguồn thứ tự thời gian.
_seq_source = getattr(engine, "last_sequence_source", None)
if _seq_source:
    if _seq_source == "FLOW_START_MILLISECONDS":
        st.caption(
            f"Biến trình tự: `{_seq_source}`, tức mốc thời gian quan sát được "
            "trong chính tệp đầu vào."
        )
    else:
        st.caption(
            f"Biến trình tự: {_seq_source}. Tệp này không có cột "
            "`FLOW_START_MILLISECONDS`, nên mô hình chỉ biết thứ tự trước sau "
            "giữa các luồng, không biết chúng cách nhau bao lâu."
        )


# Trạng thái trọng số và đặc trưng đầu vào.
if engine.last_predict_used_real_model is None:
    _badge_real = engine.is_model_loaded
else:
    _badge_real = engine.last_predict_used_real_model

mode_badge_class = "active" if _badge_real else "danger"
if _badge_real and engine.last_missing_features:
    mode_badge_text = f"weights loaded &middot; {len(engine.last_missing_features)} feature columns missing, zero-filled"
elif _badge_real:
    mode_badge_text = "weights loaded"
elif engine.is_model_loaded:
    mode_badge_text = "demo mode &middot; inference failed, see log"
else:
    mode_badge_text = "demo mode &middot; not real inference"


# Dải mật độ tấn công theo thứ tự luồng.
if analysis_mode == "Batch Analysis":
    _timeline_source = (
        selected_scenario if data_source != "Upload CSV File" else "Uploaded File"
    )
else:
    _timeline_source = (
        (rt_data_label or "Uploaded stream") if rt_source != "Upload Stream File" else "Uploaded stream"
    )
st.markdown(
    render_capture_timeline_html(
        df_result if analysis_mode == "Batch Analysis" else None,
        source_label=_timeline_source,
        status_text=mode_badge_text,
        status_ok=(mode_badge_class != "danger"),
    ),
    unsafe_allow_html=True,
)


def cham_tren_nhan_that(frame):
    # So sánh dự đoán với nhãn gốc, nếu có.
    if frame is None or len(frame) == 0 or "prediction" not in frame.columns:
        return None
    cot_nhan = next((c for c in frame.columns
                     if str(c).strip().lower() == "label"), None)
    if cot_nhan is None:
        return None
    that = pd.to_numeric(frame[cot_nhan], errors="coerce")
    doan = pd.to_numeric(frame["prediction"], errors="coerce")
    dung_duoc = that.notna() & doan.notna()
    if not bool(dung_duoc.any()):
        return None
    that = that[dung_duoc] == 1
    doan = doan[dung_duoc] == 1
    tp = int((that & doan).sum())
    fn = int((that & ~doan).sum())
    fp = int((~that & doan).sum())
    tn = int((~that & ~doan).sum())
    tong = tp + fn + fp + tn
    return {
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "far": (fp / (fp + tn)) if (fp + tn) else None,
        "acc": (tp + tn) / tong if tong else None,
        "fp": fp,
        "fn": fn,
        "tong": tong,
    }


if analysis_mode == "Batch Analysis":
    _diem = cham_tren_nhan_that(df_result)
    if _diem is not None:
        def _ba_chu_so(v):
            return "&mdash;" if v is None else f"{v * 100:.3f}%"

        st.markdown(
            f"""
            <div class="scored-panel">
              <div class="scored-head">scored against this file's own ground-truth labels</div>
              <div class="capture-stats">
                <div class="stat"><span class="stat-num">{_ba_chu_so(_diem['recall'])}</span><span class="stat-lab">attack recall</span></div>
                <div class="stat"><span class="stat-num is-threat">{_ba_chu_so(_diem['far'])}</span><span class="stat-lab">false alarm rate</span></div>
                <div class="stat"><span class="stat-num">{_ba_chu_so(_diem['acc'])}</span><span class="stat-lab">accuracy</span></div>
                <div class="stat"><span class="stat-num is-threat">{_diem['fp']:,}</span><span class="stat-lab">benign flows misflagged</span></div>
                <div class="stat"><span class="stat-num is-threat">{_diem['fn']:,}</span><span class="stat-lab">attacks missed</span></div>
              </div>
              <div class="scored-note">Computed on the {_diem['tong']:,} labelled flows of this demo file.
              These are not a capability estimate on unlabelled traffic, and the restore-from-checkpoint
              figures read optimistically because the saved memory table was written after the model had
              already traversed this test split.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if analysis_mode == "Batch Analysis" and len(df_result) > 0:
    total_flows = len(df_result)
    total_attacks = int((df_result["prediction"] == 1).sum()) if "prediction" in df_result.columns else 0
    total_benign = total_flows - total_attacks
    attack_rate = total_attacks / total_flows * 100 if total_flows > 0 else 0

    # Tổng hợp hàng đợi các luồng cần xem xét.
    _attrs = getattr(df_result, "attrs", {}) or {}
    _n_uncertain = int(_attrs.get("review_n_uncertain", 0))
    _overflow = int(_attrs.get("review_overflow", 0))

    review_count = getattr(engine, "last_needs_review_count", 0)
    if review_count > 0:
        review_rate = review_count / total_flows * 100 if total_flows > 0 else 0
        _unc_rate = _n_uncertain / total_flows * 100 if total_flows > 0 else 0
        if _overflow > 0:
            _review_title = (
                f"Hàng đợi xem xét — {review_count:,} dòng, còn {_overflow:,} chưa tới lượt "
                f"(tổng {_n_uncertain:,} dòng dưới ngưỡng tin cậy = {format_exact_percent(_unc_rate / 100.0)})"
            )
        else:
            _review_title = (
                f"Hàng đợi xem xét — {review_count:,} dòng "
                f"({format_exact_percent(_unc_rate / 100.0)} số flow dưới ngưỡng tin cậy, đã phủ hết)"
            )
        with st.expander(
                _review_title,
                expanded=False,
            ):
            _capacity = int(_attrs.get("review_capacity", 0))
            _thr = float(_attrs.get("review_threshold", 0.20))
            _overflow_html = (
                f'<div class="review-note is-warn">Quá tải: {_overflow:,} dòng dưới ngưỡng tin cậy '
                f'chưa tới lượt xem trong lần chạy này. Nâng sức xử lý hoặc chấp nhận rằng phần '
                f'tồn đọng không được ai kiểm tra.</div>'
            ) if _overflow > 0 else ""

            _prof = build_review_profile(
                df_result,
                missing_features=getattr(engine, "last_missing_features", []),
                review_budget_pct=review_budget,
            )
            _profile_html = "".join(
                f'<div class="review-rule is-{lvl}">'
                f'<span class="rr-key" style="grid-column:1/3;">{title}</span>'
                f'<span class="rr-why">{body}</span></div>'
                for lvl, title, body in _prof["notes"]
            ) or '<div class="review-note">Chưa đủ dữ liệu để nhận xét.</div>'

            st.markdown(
                f"""
                <div class="review-panel">
                  <div class="review-head">Mô hình quyết định cái gì cần xem, hạn mức chỉ quyết định xem được bao nhiêu</div>
                  <div class="review-rules">
                    <div class="review-rule">
                      <span class="rr-val">{_n_uncertain:,}</span>
                      <span class="rr-key">dòng dưới ngưỡng tin cậy</span>
                      <span class="rr-why">mô hình tự xác định theo ngưỡng margin {_thr:.0%}, không phụ thuộc hạn mức</span>
                    </div>
                    <div class="review-rule">
                      <span class="rr-val">{review_budget}%</span>
                      <span class="rr-key">sức xử lý đã đặt</span>
                      <span class="rr-why">tương đương {_capacity:,} dòng mỗi lần chạy</span>
                    </div>
                    <div class="review-rule">
                      <span class="rr-val">margin</span>
                      <span class="rr-key">tiêu chí xếp hạng chính</span>
                      <span class="rr-why">chênh lệch xác suất giữa lớp nhất và lớp nhì, càng nhỏ càng phân vân</span>
                    </div>
                    <div class="review-rule">
                      <span class="rr-val">&gt; {MISSING_FEATURE_RATIO_THRESHOLD:.0%}</span>
                      <span class="rr-key">tỷ lệ đặc trưng thiếu</span>
                      <span class="rr-why">vượt ngưỡng này thì cả file vào hàng đợi bất kể hạn mức</span>
                    </div>
                  </div>
                  <div class="review-head" style="margin-top:16px;">Đặc điểm bộ dữ liệu đang xem</div>
                  {_profile_html}
                  <div class="review-note">
                    Đây là tín hiệu <strong>mô hình không chắc</strong> hoặc <strong>đầu vào không đầy đủ</strong>,
                    không phải tín hiệu mô hình sai. Lúc suy luận không có nhãn thật nên không thể biết đúng hay sai.
                  </div>
                  {_overflow_html}
                  <div class="review-note is-warn">
                    Giới hạn đã biết: dự đoán bỏ sót có thể mang confidence rất cao, nên tiêu chí
                    margin không chắc bắt được nhóm đó.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            review_df = getattr(engine, "last_needs_review_df", None)
            if review_df is not None and len(review_df) > 0:
                display_cols = [c for c in [
                    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
                    "prediction_label", "confidence", "uncertainty_margin", "uncertainty_entropy",
                    "missing_features_ratio",
                ] if c in review_df.columns]
                st.markdown(render_df_html(review_df[display_cols], "Dòng cần xem xét",
                                           icon="rule", max_rows=200), unsafe_allow_html=True)
            else:
                st.caption("Không có dữ liệu chi tiết để hiển thị.")

st.markdown("<br>", unsafe_allow_html=True)


# Ba khu vực chính của dashboard.
tab1, tab2, tab3 = st.tabs([
    "Real-time Traffic",
    "Model Metrics",
    "XAI Explainer",
])


# Phân tích dữ liệu theo lô hoặc theo luồng.
with tab1:
    if analysis_mode == "Batch Analysis":
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.plotly_chart(
                plot_traffic_timeline(df_result),
                width="stretch",
                key="traffic_timeline",
            )

        with col_right:
            st.plotly_chart(
                plot_attack_distribution(df_result),
                width="stretch",
                key="attack_dist",
            )

        # Hiển thị các luồng bị gắn cờ.
        alerts_df = engine.get_alerts(df_result)
        if len(alerts_df) > 0:
            st.markdown(
                render_alert_table_html(alerts_df),
                unsafe_allow_html=True,
            )
        else:
            st.success("Không phát hiện tấn công nào trong dữ liệu hiện tại.")

        # Hiển thị dữ liệu đầu vào và kết quả.
        with st.expander("Raw Data (click to expand)", expanded=False):
            display_cols = ["prediction", "confidence", "anomaly_score"]
            extra_cols = [c for c in display_cols if c in df_result.columns]

            cols_lower = {str(c).strip().lower(): c for c in df_result.columns}
            show_cols = []
            for target in ["ipv4_src_addr", "l4_src_port", "ipv4_dst_addr", "l4_dst_port", "protocol", "attack", "label"]:
                if target in cols_lower:
                    show_cols.append(cols_lower[target])
            show_cols.extend(extra_cols)

            available_show = [c for c in show_cols if c in df_result.columns]
            if available_show:
                st.markdown(render_df_html(df_result[available_show], "Dữ liệu thô",
                                           icon="table_rows"), unsafe_allow_html=True)
            else:
                st.markdown(render_df_html(df_result, "Dữ liệu thô", icon="table_rows"),
                            unsafe_allow_html=True)

    else:
        # Xử lý một lô của luồng đang phát.
        rt_is_active = st.session_state.get("rt_active", False)
        rt_is_paused = st.session_state.get("rt_paused", False)
        rt_is_complete = st.session_state.get("rt_complete", False)

        if rt_is_active and not rt_is_paused and not rt_is_complete and len(rt_full_df) > 0:
            idx = st.session_state["rt_index"]
            total = len(rt_full_df)

            if idx < total:
                end_idx = min(idx + rt_batch_size, total)
                batch_df = rt_full_df.iloc[idx:end_idx].copy()

                # Chấm lô dữ liệu hiện tại.
                engine.review_budget_pct = float(review_budget)
                batch_result, accumulated = engine.predict_batch(
                    batch_df,
                    st.session_state["rt_accumulated"]
                )

                st.session_state["rt_accumulated"] = accumulated
                st.session_state["rt_index"] = end_idx
                st.session_state["rt_tick"] += 1

                # Lưu cảnh báo của lô.
                if "prediction" in batch_result.columns:
                    attack_rows = batch_result[batch_result["prediction"] == 1]
                    cols_lower_rt = {str(c).strip().lower(): c for c in batch_result.columns}
                    src_col_rt = cols_lower_rt.get("ipv4_src_addr", cols_lower_rt.get("srcip", "IPV4_SRC_ADDR"))
                    dst_col_rt = cols_lower_rt.get("ipv4_dst_addr", cols_lower_rt.get("dstip", "IPV4_DST_ADDR"))
                    attack_col_rt = cols_lower_rt.get("prediction_label", cols_lower_rt.get("attack", None))

                    KHONG_CO = "không có trong dữ liệu"
                    for _, row in attack_rows.iterrows():
                        s_ip = str(row[src_col_rt]) if src_col_rt and src_col_rt in row.index else KHONG_CO
                        d_ip = str(row[dst_col_rt]) if dst_col_rt and dst_col_rt in row.index else KHONG_CO
                        a_type = str(row[attack_col_rt]) if attack_col_rt and attack_col_rt in row.index else str(row.get("prediction_label", "Threat"))

                        st.session_state["rt_alerts"].append({
                            "batch": st.session_state["rt_tick"],
                            "src_ip": s_ip,
                            "dst_ip": d_ip,
                            "attack_type": a_type,
                            "confidence": float(row["confidence"]) if "confidence" in row.index else None,
                        })

                # Lưu tỷ lệ tấn công theo lô.
                batch_total = len(batch_result)
                batch_attacks = int((batch_result["prediction"] == 1).sum()) if "prediction" in batch_result.columns else 0
                batch_rate = batch_attacks / batch_total * 100 if batch_total > 0 else 0
                st.session_state["rt_batch_history"].append({
                    "batch_idx": st.session_state["rt_tick"],
                    "total": batch_total,
                    "attacks": batch_attacks,
                    "attack_rate": batch_rate,
                })

                if end_idx >= total:
                    st.session_state["rt_active"] = False
                    st.session_state["rt_complete"] = True
            else:
                st.session_state["rt_active"] = False
                st.session_state["rt_complete"] = True

        acc_df = st.session_state.get("rt_accumulated", pd.DataFrame())

        if len(acc_df) > 0:
            acc_total = len(acc_df)
            acc_attacks = int((acc_df["prediction"] == 1).sum()) if "prediction" in acc_df.columns else 0
            acc_benign = acc_total - acc_attacks
            acc_rate = acc_attacks / acc_total * 100 if acc_total > 0 else 0

            is_streaming = rt_is_active and not rt_is_paused
            st.markdown(
                create_realtime_metrics_html(acc_total, acc_benign, acc_attacks, acc_rate, is_streaming),
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.plotly_chart(
                    plot_realtime_timeline(acc_df, window_size=rt_batch_size),
                    width="stretch",
                    key="rt_timeline",
                )

            with col_right:
                st.plotly_chart(
                    plot_realtime_attack_rate(st.session_state.get("rt_batch_history", [])),
                    width="stretch",
                    key="rt_attack_rate",
                )

            st.markdown(
                render_live_alert_feed_html(st.session_state.get("rt_alerts", [])),
                unsafe_allow_html=True,
            )

            if rt_is_complete:
                st.markdown(f"""
                <div class="realtime-header">
                    <p style="color:var(--text-main);font-weight:700;font-size:1.1rem;margin:0;">
                        <span class="material-symbols-outlined">task_alt</span> Streaming Complete
                    </p>
                    <p style="color:var(--text-muted);font-family:monospace;font-size:0.9rem;margin:4px 0 0;">
                        Processed <strong>{acc_total:,}</strong> flows in 
                        <strong>{st.session_state.get('rt_tick', 0)}</strong> batches
                        &nbsp;|&nbsp;
                        Detected <strong style="color:#ef4444;">{acc_attacks:,}</strong> attacks
                        ({format_exact_percent(acc_rate / 100.0)})
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Raw Accumulated Data", expanded=False):
                display_cols = ["prediction", "confidence", "anomaly_score"]
                extra_cols = [c for c in display_cols if c in acc_df.columns]
                cols_lower = {str(c).strip().lower(): c for c in acc_df.columns}
                show_cols = []
                for target in ["ipv4_src_addr", "l4_src_port", "ipv4_dst_addr", "l4_dst_port", "protocol", "attack", "label"]:
                    if target in cols_lower:
                        show_cols.append(cols_lower[target])
                show_cols.extend(extra_cols)
                available_show = [c for c in show_cols if c in acc_df.columns]
                if available_show:
                    st.markdown(render_df_html(acc_df[available_show].tail(50),
                                               "Dữ liệu tích luỹ", icon="table_rows"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(render_df_html(acc_df.tail(50), "Dữ liệu tích luỹ",
                                               icon="table_rows"), unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="realtime-header">
                <p style="color:var(--text-muted);font-weight:600;font-size:1rem;margin:0;">
                    Nhấn <strong style="color:#f87171;">Start</strong> ở thanh bên trái để bắt đầu mô phỏng.
                </p>
                <p style="color:var(--text-muted);font-family:monospace;font-size:0.85rem;margin:4px 0 0;">
                    Data: {rt_data_file} &nbsp;|&nbsp; Batch: {rt_batch_size} flows &nbsp;|&nbsp; Speed: {rt_speed}s/batch
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Phát lô tiếp theo khi stream đang chạy.
        if rt_is_active and not rt_is_paused and not rt_is_complete:
            time.sleep(rt_speed)
            st.rerun()


# Đối sánh chỉ số các mô hình.
with tab2:
    all_metrics = get_all_metrics_for_comparison()

    # Chỉ số của mô hình đang chọn.
    current_metrics = load_saved_metrics(model_version)
    if current_metrics:
        st.markdown(f"### {model_version} — Performance Gauges")
        st.plotly_chart(
            plot_metrics_gauges(current_metrics),
            width="stretch",
            key="gauges",
        )

    st.markdown("---")

    # So sánh giữa các mô hình và mốc tham chiếu.
    st.markdown("### Multi-Model Comparison")
    st.plotly_chart(
        plot_metrics_comparison(all_metrics),
        width="stretch",
        key="comparison",
    )
    st.markdown(build_metrics_comparison_legend_html(all_metrics), unsafe_allow_html=True)

    # Ma trận nhầm lẫn của mô hình đa lớp.
    v3_metrics = load_saved_metrics("TGN-NIDS Multiclass")
    if v3_metrics:
        st.markdown("#### TGN-NIDS Multiclass")
        st.plotly_chart(
            plot_confusion_matrix(v3_metrics, title="TGN-NIDS Multiclass"),
            width="stretch",
            key="cm_v3",
        )

    # Bảng đối sánh chi tiết.
    with st.expander("Detailed Metrics Table", expanded=False):
        table_data = []
        for version, metrics in all_metrics.items():
            row = {"Model": version}
            for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_macro", "fpr", "auc_roc"]:
                row[key.upper()] = format_exact_percent(metrics.get(key))
            table_data.append(row)

        st.markdown(render_df_html(pd.DataFrame(table_data), "Chỉ số chi tiết",
                                   icon="analytics", max_rows=50), unsafe_allow_html=True)


# Phân tích XAI theo đồ thị lân cận và đặc trưng.
with tab3:
    st.markdown("### Explainable AI — Ego-Network & Feature Importance")

    if analysis_mode == "Batch Analysis":
        xai_df = df_result
    else:
        xai_df = st.session_state.get("rt_accumulated", pd.DataFrame())

    if len(xai_df) == 0:
        st.info("Chưa có dữ liệu để phân tích. Chạy Batch Analysis hoặc bắt đầu Realtime Streaming trước.")
    else:
        # Chọn endpoint để phân tích.
        xai_cols_lower = {str(c).strip().lower(): c for c in xai_df.columns}
        xai_src_col = xai_cols_lower.get("ipv4_src_addr", xai_cols_lower.get("srcip", xai_cols_lower.get("src_ip")))
        xai_dst_col = xai_cols_lower.get("ipv4_dst_addr", xai_cols_lower.get("dstip", xai_cols_lower.get("dst_ip")))
        if xai_src_col in xai_df.columns and xai_dst_col in xai_df.columns:
            xai_ips = sorted(set(xai_df[xai_src_col].astype(str)) | set(xai_df[xai_dst_col].astype(str)))
        else:
            xai_ips = []
        ip_options = {ip: ip for ip in xai_ips}

        if ip_options:
            _sel_key = "xai_target_ip_" + "__".join(str(part) for part in analysis_context)
            _pending = st.session_state.pop("xai_pending_target", None)
            if _pending is not None and _pending in ip_options:
                st.session_state[_sel_key] = _pending

            selected_target_ip = st.selectbox(
                "Select Target IP for Analysis" if "endpoint_identity_source" not in xai_df.columns or (xai_df["endpoint_identity_source"] == "observed").all() else "Select Anonymous Endpoint for Analysis",
                options=list(ip_options.keys()),
                key=_sel_key,
                help="Chọn endpoint cần điều tra để xem các endpoint liên quan và đặc trưng nào khiến mô hình gắn cờ.",
            )
            if "endpoint_identity_source" in xai_df.columns and (xai_df["endpoint_identity_source"] != "observed").any():
                st.caption(
                    "Dataset không có IP thật; endpoint hiển thị chỉ là định danh tổng hợp từ "
                    "port hoặc flow, không phải attacker IP. Endpoint nguồn và đích nằm ở hai "
                    "không gian tên tách biệt nên đồ thị chỉ gồm các cặp rời rạc: tăng độ rộng "
                    "vùng lân cận sẽ gần như không mở rộng thêm. Phân tích lân cận chỉ có ý "
                    "nghĩa với dữ liệu có endpoint thật."
                )
            target_node = engine.ip_to_id.get(selected_target_ip, 0)
        else:
            selected_target_ip = None
            target_node = 0
            st.info("Không tìm thấy ánh xạ endpoint. Dùng Node 0 làm mặc định.")

        # Chạy phân tích XAI.
        if st.button("Run XAI Analysis", type="primary"):
            with st.spinner("Analyzing network topology and feature importance..."):
                xai_results = engine.get_xai_results(
                    xai_df,
                    target_node_id=target_node,
                    k_hops=k_hops,
                    target_ip=selected_target_ip,
                )

            st.session_state["xai_results"] = xai_results

        # Hiển thị đồ thị và độ quan trọng đặc trưng.
        if "xai_results" in st.session_state:
            xai = st.session_state["xai_results"]
            subgraph = xai.get("subgraph", {})
            importance = xai.get("feature_importance", {})
            ip_map = xai.get("ip_map", {})

            # Lọc cạnh theo ngưỡng điểm bất thường.
            _nguong = st.session_state.get("xai_score_filter")
            _nguong = float(_nguong) if _nguong else 0.0
            _canh = loc_canh_theo_nguong(subgraph.get("edges", []), _nguong)
            _da_loc = len(_canh) != len(subgraph.get("edges", []))
            if _da_loc:
                subgraph = dict(
                    subgraph, edges=_canh, num_edges=len(_canh),
                    num_pairs=len({(e.get("src_ip"), e.get("dst_ip"))
                                   for e in _canh}))

            _n_shown = subgraph.get("num_edges", 0)
            _n_total = subgraph.get("num_edges_total", _n_shown)
            _n_pairs = subgraph.get("num_pairs", 0)
            _edge_txt = (
                f"{_n_pairs} kết nối, gộp từ {_n_shown}/{_n_total} luồng "
                "(ưu tiên luồng bị gắn cờ)"
                if subgraph.get("edges_truncated")
                else f"{_n_pairs} kết nối, gộp từ {_n_shown} luồng"
            )
            if _da_loc:
                _edge_txt += f", còn lại sau ngưỡng lọc {_nguong:g}"

            _lech = []
            if subgraph.get("co_nhan_that"):
                if subgraph.get("so_bo_sot"):
                    _lech.append(f"{subgraph['so_bo_sot']} luồng mô hình bỏ sót")
                if subgraph.get("so_bao_nham"):
                    _lech.append(f"{subgraph['so_bao_nham']} luồng báo nhầm")
            _lech_txt = ("&nbsp;&nbsp;|&nbsp;&nbsp;<strong>So với nhãn thật:</strong> "
                         + ", ".join(_lech)) if _lech else ""

            _sau = int(subgraph.get("max_hop", 0) or 0)
            _sau_txt = (f" (dữ liệu chỉ sâu {_sau} bước)"
                        if _sau < int(subgraph.get("k_hops", k_hops)) else "")

            c_node_id = subgraph.get("original_center_node", subgraph.get("center_node", 0))
            c_node_label = subgraph.get("center_ip", ip_map.get(c_node_id, f"Node {c_node_id}"))

            st.markdown(f"""
            <div class="section-card">
                <p style="color: var(--text-main); margin: 0; font-family: monospace; font-size: 0.9rem;">
                    <strong>Center Node:</strong> {_esc(c_node_label)}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <strong>Đồ thị con:</strong> {subgraph.get('num_nodes', 0)} endpoint, {_edge_txt}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <strong>Vùng lân cận:</strong> {subgraph.get('k_hops', k_hops)} bước{_sau_txt}{_lech_txt}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Điều khiển bố cục và bộ lọc đồ thị.
            _ctl1, _ctl2, _ctl3 = st.columns([3, 2, 3])
            with _ctl1:
                xai_layout_mode = st.radio(
                    "Bố cục",
                    ["Gọn gàng (2 cột)", "Mở rộng toàn màn hình"],
                    horizontal=True,
                    key="xai_layout_choice",
                    label_visibility="collapsed",
                )
            with _ctl2:
                enable_force_physics = st.toggle("Mô phỏng vật lý", value=False, key="xai_physics_toggle",
                                                 help="Bật lực hút tự do để các node tự giãn cách theo cụm.")
            with _ctl3:
                min_score_filter = st.slider("Lọc điểm bất thường tối thiểu", 0.0, 1.0, 0.0, 0.05,
                                             key="xai_score_filter",
                                             help="Ẩn các luồng bình thường có điểm dưới ngưỡng để giảm rối mắt.")

            is_full_width = (xai_layout_mode == "Mở rộng toàn màn hình")
            canvas_height = 650 if is_full_width else 520

            if is_full_width:
                st.markdown("#### Đồ thị lân cận (Bản đồ không gian mở rộng)")
                st.markdown(
                    """
                    <div class="graph-legend">
                      <span class="gl-item"><span class="gl-dot" style="background:var(--neon-cyan);width:14px;height:14px;"></span>mục tiêu</span>
                      <span class="gl-item"><span class="gl-dot" style="background:var(--neon-amber);"></span>cách 1 bước</span>
                      <span class="gl-item"><span class="gl-dot" style="background:#8a8175;width:8px;height:8px;"></span>cách 2 bước</span>
                      <span class="gl-sep"></span>
                      <span class="gl-item"><span class="gl-line" style="background:#dc2626;height:3px;"></span>mô hình gắn cờ tấn công</span>
                      <span class="gl-item"><span class="gl-line" style="background:#4a3a30;"></span>mô hình không gắn cờ</span>
                      <span class="gl-item"><span class="gl-line" style="background:transparent;height:0;border-top:2px dashed #c8c2b8;border-radius:0;"></span>nhãn thật có tấn công mà mô hình bỏ sót</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                html_content = render_ego_network_html(
                    subgraph,
                    height=f"{canvas_height}px",
                    enable_physics=enable_force_physics,
                )
                if html_content.startswith("<p"):
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    import streamlit.components.v1 as components
                    components.html(html_content, height=canvas_height, scrolling=False)

                st.markdown("#### Độ quan trọng đặc trưng")
                _imp_meta = xai.get("feature_importance_meta", {})
                if importance:
                    st.plotly_chart(
                        plot_feature_importance(importance, top_n=15),
                        width="stretch",
                        key="feat_importance_full",
                    )
                else:
                    st.warning(_imp_meta.get("reason", "Chưa đo được độ quan trọng đặc trưng."))
            else:
                col_graph, col_feat = st.columns([3, 2])

                with col_graph:
                    st.markdown("#### Đồ thị lân cận")
                    st.markdown(
                        """
                        <div class="graph-legend">
                          <span class="gl-item"><span class="gl-dot" style="background:var(--neon-cyan);width:14px;height:14px;"></span>mục tiêu</span>
                          <span class="gl-item"><span class="gl-dot" style="background:var(--neon-amber);"></span>cách 1 bước</span>
                          <span class="gl-item"><span class="gl-dot" style="background:#8a8175;width:8px;height:8px;"></span>cách 2 bước</span>
                          <span class="gl-sep"></span>
                          <span class="gl-item"><span class="gl-line" style="background:#dc2626;height:3px;"></span>mô hình gắn cờ tấn công</span>
                          <span class="gl-item"><span class="gl-line" style="background:#4a3a30;"></span>mô hình không gắn cờ</span>
                          <span class="gl-item"><span class="gl-line" style="background:transparent;height:0;border-top:2px dashed #c8c2b8;border-radius:0;"></span>nhãn thật có tấn công mà mô hình bỏ sót</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    _top = subgraph.get("top_attacks", [])
                    _n_direct = subgraph.get("num_direct_total", 0)
                    _n_key = len(subgraph.get("key_neighbours", []))
                    if _top:
                        st.markdown(
                            f'<div class="top-attacks-head">Đáng chú ý quanh mục tiêu &middot; '
                            f'đồ thị vẽ {_n_key}/{_n_direct} endpoint nối trực tiếp</div>',
                            unsafe_allow_html=True,
                        )
                        for _i, _atk in enumerate(_top):
                            _c1, _c2 = st.columns([5, 1])
                            with _c1:
                                st.markdown(
                                    f'<div class="top-attack-row">'
                                    f'<span class="ta-peer">{_esc(_atk["peer"])}</span>'
                                    f'<span class="ta-type">{_esc(_atk["attack_type"])}</span>'
                                    f'<span class="ta-att">{_fmt_score(_atk.get("anomaly_score"))}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            with _c2:
                                if st.button("Phân tích", key=f"jump_{_i}_{_atk['peer']}",
                                             help=f"Chuyển mục tiêu XAI sang {_atk['peer']}"):
                                    st.session_state["xai_pending_target"] = _atk["peer"]
                                    st.session_state.pop("xai_results", None)
                                    st.rerun()

                    # Tạo đồ thị tương tác.
                    html_content = render_ego_network_html(
                        subgraph,
                        height=f"{canvas_height}px",
                        enable_physics=enable_force_physics,
                    )

                    if html_content.startswith("<p"):
                        st.markdown(html_content, unsafe_allow_html=True)
                        st.markdown("**Nodes:**")
                        for node in subgraph.get("nodes", []):
                            icon = "[*]" if node.get("is_center", False) else "[ ]"
                            nid = node.get("original_id", node.get("id", 0))
                            ip = ip_map.get(nid, f"Node {nid}")
                            st.markdown(f"  {icon} {ip}")

                        st.markdown("**Edges:**")
                        for edge in subgraph.get("edges", [])[:15]:
                            s_id = edge.get("source", 0)
                            t_id = edge.get("target", 0)
                            src_ip = edge.get("src_ip", ip_map.get(s_id, f"Node {s_id}"))
                            dst_ip = edge.get("dst_ip", ip_map.get(t_id, f"Node {t_id}"))
                            status = "ATTACK" if edge.get("is_anomaly", False) else "Normal"
                            st.markdown(f"  {src_ip} → {dst_ip} [{status}]")
                    else:
                        st.iframe(html_content, height=canvas_height)

                with col_feat:
                    st.markdown("#### Độ quan trọng đặc trưng")
                    _imp_meta = xai.get("feature_importance_meta", {})
                    if importance:
                        st.caption(
                            "Đo bằng phép triệt tiêu thuộc tính (zero-out): lần lượt đặt "
                            "từng cột về 0 rồi đo độ lệch xác suất tấn công trên "
                            f"{_imp_meta.get('n_edges_measured', 0):,} luồng của đồ thị con. "
                            f"{_imp_meta.get('n_features_nonzero', 0)}/"
                            f"{_imp_meta.get('n_features_probed', 0)} đặc trưng có ảnh hưởng khác 0."
                        )
                        _filled = _imp_meta.get("filled_features", [])
                        if _filled:
                            st.caption(
                                f":orange[Lưu ý: {len(_filled)} đặc trưng trong bảng này "
                                f"({_imp_meta.get('pct_from_filled', 0)}% tổng độ quan trọng) "
                                "không có trong file đang xem và đã được điền giá trị thay thế. "
                                "Phần đóng góp của chúng phản ánh giá trị điền, không phải lưu "
                                f"lượng quan sát được: {', '.join(_filled)}.]"
                            )
                        st.plotly_chart(
                            plot_feature_importance(importance, top_n=15),
                            width="stretch",
                            key="feat_importance",
                        )
                    else:
                        st.warning(_imp_meta.get("reason", "Chưa đo được độ quan trọng đặc trưng."))

            # Liệt kê các cạnh trong đồ thị con.
            with st.expander(":material/search: Chi tiết cạnh (điểm bất thường theo từng flow)", expanded=True):
                ego_edges = subgraph.get("edges", [])
                if ego_edges:
                    sorted_edges = sorted(
                        ego_edges,
                        key=lambda x: (not x.get("is_anomaly", False), -float(x.get("anomaly_score") or 0.0))
                    )
                    _rows = []
                    for e in sorted_edges[:50]:
                        s_ip = str(e.get("src_ip", ip_map.get(e.get("source"), f"Node-{e.get('source')}")))
                        d_ip = str(e.get("dst_ip", ip_map.get(e.get("target"), f"Node-{e.get('target')}")))
                        is_atk = e.get("is_anomaly", False)
                        atk_type = str(e.get("attack_type", "Threat" if is_atk else "Normal Flow"))
                        _sc = e.get("anomaly_score")
                        attn_txt = format_exact_number(_sc)
                        _icon = "gpp_maybe" if is_atk else "check_circle"
                        _cls = " is-atk" if is_atk else ""
                        _label = atk_type if is_atk else "Normal flow"
                        _src, _src_anon = _format_endpoint(s_ip)
                        _dst, _dst_anon = _format_endpoint(d_ip)
                        _rows.append(
                            f'<tr class="alert-row">'
                            f'<td class="c-endpoint{" is-anon" if _src_anon else ""}" title="{_esc(s_ip)}">{_esc(_src)}</td>'
                            f'<td class="c-endpoint{" is-anon" if _dst_anon else ""}" title="{_esc(d_ip)}">{_esc(_dst)}</td>'
                            f'<td class="c-attack{_cls}">'
                            f'<span class="material-symbols-outlined edge-ico">{_icon}</span>{_esc(_label)}</td>'
                            f'<td class="c-conf">{attn_txt}</td>'
                            f"</tr>"
                        )
                    st.markdown(
                        '<div class="alert-feed"><div class="alert-feed-head">'
                        '<span class="material-symbols-outlined">lan</span>'
                        f'<span>Chi tiết cạnh</span><span class="alert-feed-count">{len(sorted_edges)} luồng</span>'
                        '</div><div class="alert-feed-body"><table class="alert-table"><thead><tr>'
                        '<th class="c-endpoint" title="Bên khởi tạo flow.">Nguồn</th>'
                        '<th class="c-endpoint" title="Bên nhận flow.">Đích</th>'
                        '<th class="c-attack" title="Lớp tấn công mô hình dự đoán, không phải nhãn thật.">Loại</th>'
                        '<th class="c-conf" title="Điểm bất thường do đầu phân loại sinh ra cho flow này (0-1). Đây không phải trọng số attention của lớp TransformerConv.">Điểm bất thường</th>'
                        '</tr></thead><tbody>' + "".join(_rows) + "</tbody></table></div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Đồ thị con này không có cạnh nào để hiển thị.")


# Thông tin đồ án.
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 0.75rem; padding: 10px 0;">
    <strong>TGN-NIDS Dashboard</strong> — Đồ án tốt nghiệp UIT-CITD 2026
    <br>
    Phạm Văn Cường (25410027) &amp; Đặng Thiên Phước (25410111)
    <br>
    Giảng viên hướng dẫn: TS. Phan Thế Duy
</div>
""", unsafe_allow_html=True)
