"""Công cụ giải thích đặc trưng và đồ thị con của mô hình TGN-NIDS."""

__all__ = ["GNNExplainer", "TGNExplainer", "SubgraphExtractor"]

# Chỉ nạp PyTorch khi một lớp giải thích thực sự được sử dụng.
_MODULE_OF = {
    "GNNExplainer": "gnn_explainer",
    "TGNExplainer": "gnn_explainer",
    "SubgraphExtractor": "subgraph_extractor",
}


def __getattr__(name):
    """Nạp lớp công khai theo yêu cầu."""
    module_name = _MODULE_OF.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)

    # Lưu lại lớp đã nạp để các lần truy cập sau không phải import lại.
    globals()[name] = value
    return value


def __dir__():
    """Liệt kê các lớp công khai cho `dir()` và trình soạn thảo."""
    return sorted(__all__)
