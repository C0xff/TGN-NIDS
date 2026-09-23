"""Bảng màu và CSS dùng chung cho hai giao diện sáng, tối của dashboard."""

# Màu dùng cho hai chế độ giao diện.
THEME_TOKENS = {
    "Tối": {
        "is_dark": True,
        "bg-base": "#0a0a0a",
        "bg-surface": "rgba(26, 26, 26, 0.78)",
        "bg-card": "rgba(32, 32, 32, 0.68)",
        "bg-card-hover": "rgba(44, 44, 44, 0.85)",
        "border-subtle": "rgba(255, 255, 255, 0.10)",
        "border-accent": "rgba(255, 255, 255, 0.30)",
        "border-danger": "rgba(209, 59, 59, 0.55)",
        "neon-cyan": "#ffffff",
        "neon-emerald": "#b8b8b8",
        "neon-amber": "#d6d6d6",
        "neon-rose": "#d13b3b",
        "neon-purple": "#9a9a9a",
        "text-main": "#e6e6e6",
        "text-muted": "#8c8c8c",
        "table-dim": "#b5b5b5",
        "text-highlight": "#ffffff",
        "page-bg": "radial-gradient(circle at 15% 15%, #1c1c1c 0%, #101010 45%, #060606 100%)",
        "panel-bg": "linear-gradient(135deg, rgba(30, 30, 30, 0.72) 0%, rgba(18, 18, 18, 0.85) 100%)",
        "sidebar-bg": "linear-gradient(180deg, rgba(22, 22, 22, 0.98) 0%, rgba(9, 9, 9, 1.0) 100%)",
        "rib-top": "#f97316", "rib-bottom": "#9a3412",
        "chart_bg": "#0a0a0a",
        # Màu biểu đồ lưu lượng bình thường và tấn công.
        "chart_benign": "#6b8e9f", "chart_attack": "#f97316",
        "chart_grid": "#2a2a2a", "chart_text": "#c8c8c8",
        "chart_template": "plotly_dark",
        "chart_seq": [[0, "#141414"], [0.25, "#3d2410"], [0.6, "#c2410c"], [1, "#fbbf24"]],
        "chart_cat": ["#f97316", "#fbbf24", "#dc2626", "#c2410c",
                      "#6b8e9f", "#eab308", "#a3a3a3", "#7d9471",
                      "#9a3412", "#d6d3d1"],
    },
    "Sáng": {
        "is_dark": False,
        "bg-base": "#f6f6f4",
        "bg-surface": "rgba(255, 255, 255, 0.86)",
        "bg-card": "rgba(255, 255, 255, 0.78)",
        "bg-card-hover": "rgba(240, 240, 238, 0.92)",
        "border-subtle": "rgba(0, 0, 0, 0.10)",
        "border-accent": "rgba(0, 0, 0, 0.32)",
        "border-danger": "rgba(179, 36, 36, 0.55)",
        "neon-cyan": "#111111",
        "neon-emerald": "#5c5c5c",
        "neon-amber": "#3d3d3d",
        "neon-rose": "#b32424",
        "neon-purple": "#6b6b6b",
        "text-main": "#1a1a1a",
        "text-muted": "#6b6b6b",
        "table-dim": "#4a4a4a",
        "text-highlight": "#000000",
        "page-bg": "radial-gradient(circle at 15% 15%, #ffffff 0%, #f3f3f1 45%, #e8e8e6 100%)",
        "panel-bg": "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(246, 246, 244, 0.95) 100%)",
        "sidebar-bg": "linear-gradient(180deg, rgba(252, 252, 251, 0.98) 0%, rgba(242, 242, 240, 1.0) 100%)",
        "rib-top": "#ea580c", "rib-bottom": "#fdba74",
        "chart_bg": "#ffffff",
        "chart_benign": "#5b7d8d", "chart_attack": "#ea580c",
        "chart_grid": "#e0e0e0", "chart_text": "#333333",
        "chart_template": "plotly_white",
        "chart_seq": [[0, "#ffffff"], [0.25, "#fde3c7"], [0.6, "#ea580c"], [1, "#7c2d12"]],
        "chart_cat": ["#f97316", "#fbbf24", "#dc2626", "#c2410c",
                      "#6b8e9f", "#eab308", "#a3a3a3", "#7d9471",
                      "#9a3412", "#d6d3d1"],
    },
}


def build_theme_css(tokens: dict) -> str:
    # Áp dụng màu của chế độ đang chọn.
    var_lines = "\n".join(
        f"    --{k}: {v} !important;"
        for k, v in tokens.items()
        if not k.startswith("chart")
        and k not in ("page-bg", "panel-bg", "sidebar-bg", "rib-top",
                      "rib-bottom", "is_dark")
    )
    return f"""
<style>
:root {{
{var_lines}
}}

.stApp {{ background: {tokens['page-bg']} !important; }}

section[data-testid="stSidebar"] {{ background: {tokens['sidebar-bg']} !important; }}

.capture-hero,
div[data-testid="stExpander"],
.alert-feed,
.metric-card {{
    background: {tokens['panel-bg']} !important;
    border-color: var(--border-subtle) !important;
}}

.rib-bar {{
    background: linear-gradient(180deg, {tokens['rib-top']} 0%, {tokens['rib-bottom']} 100%) !important;
}}

.rib-bar.is-peak {{
    background: linear-gradient(180deg, var(--neon-rose) 0%, {tokens['rib-bottom']} 100%) !important;
}}

.capture-ribbon {{ border-bottom-color: var(--border-accent) !important; }}

.alert-table th {{ background: var(--bg-card-hover) !important; }}
.alert-table .c-attack {{ color: var(--text-main) !important; }}
.alert-row.is-newest td {{ background: rgba(209, 59, 59, 0.10) !important; }}
.alert-row:hover td {{ background: var(--bg-card-hover) !important; }}

.review-rule {{ background: var(--bg-card) !important; border-left-color: var(--border-accent) !important; }}
.review-rule.is-warn {{ border-left-color: var(--neon-rose) !important; }}
.top-attack-row {{ background: var(--bg-card) !important; border-left-color: var(--neon-rose) !important; }}
.ta-type {{ color: var(--text-main) !important; }}

button[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"] {{
    color: var(--text-main) !important;
    background: var(--bg-card) !important;
    border-color: var(--border-accent) !important;
}}

button[kind="headerNoPadding"] span,
[data-testid="stIconMaterial"] {{ color: var(--text-main) !important; fill: var(--text-main) !important; }}

button[data-testid="stBaseButton-primary"],
button[kind="primary"] {{
    background: var(--text-highlight) !important;
    color: var(--bg-base) !important;
    border-color: var(--text-highlight) !important;
    box-shadow: none !important;
}}

button[data-testid="stBaseButton-primary"] *,
button[kind="primary"] * {{ color: var(--bg-base) !important; }}
</style>
"""


def build_dataframe_css(tokens: dict) -> str:
    # Đồng bộ bảng dữ liệu với giao diện.
    dark = bool(tokens.get("is_dark", False))
    return f"""
<style>
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {{
    --gdg-bg-cell: {tokens['bg-card'] if dark else '#ffffff'};
    --gdg-bg-cell-medium: {tokens['bg-card-hover']};
    --gdg-bg-header: {tokens['bg-card-hover']};
    --gdg-bg-header-hovered: {tokens['bg-card-hover']};
    --gdg-bg-header-has-focus: {tokens['bg-card-hover']};
    --gdg-text-dark: {tokens['text-main']};
    --gdg-text-medium: {tokens['table-dim']};
    --gdg-text-light: {tokens['text-muted']};
    --gdg-text-header: {tokens['table-dim']};
    --gdg-text-header-selected: {tokens['text-highlight']};
    --gdg-border-color: {tokens['border-subtle']};
    --gdg-horizontal-border-color: {tokens['border-subtle']};
    --gdg-accent-color: {tokens['neon-cyan']};
    --gdg-accent-light: {tokens['bg-card-hover']};
    --gdg-bg-bubble: {tokens['bg-card']};
    --gdg-bg-bubble-selected: {tokens['bg-card-hover']};
    --gdg-font-family: "IBM Plex Mono", monospace;
    border: 1px solid {tokens['border-subtle']} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
</style>
"""
