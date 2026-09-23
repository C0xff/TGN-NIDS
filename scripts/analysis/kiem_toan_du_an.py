"""Kiểm toán toàn bộ dự án: mã chết, chú thích, đường dẫn, dấu vết bịa, tài liệu lệch.

Chạy:  PYTHONPATH=src:apps/dashboard python scripts/kiem_toan_du_an.py

Mỗi phép kiểm ứng với một loại lỗi đã thực sự xảy ra trong dự án. Trả mã thoát 1
nếu còn phát hiện, để dùng được trong quy trình tự động.

Nguyên tắc: **mã nguồn là nguồn sự thật, tài liệu bám theo mã.** Phép kiểm cuối
đối chiếu những gì tài liệu khẳng định với những gì mã thật sự có, và khi lệch
thì mã thắng.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys
import unicodedata

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_DIRS = ("src", "apps", "scripts", "tests")

# Ký tự tiếng Việt có dấu. Dùng tập tường minh chứ không so mã điểm Unicode:
# dấu gạch dài U+2014 nằm trên mọi ngưỡng hợp lý và từng làm phép kiểm bỏ sót.
VIET_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
                 "òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")

# Từ tiếng Việt không dấu hay gặp trong chú thích.
VIET_WORDS = re.compile(
    r"\b(cua|khong|nhung|duoc|mo hinh|du lieu|ket qua|thu muc|gia tri|nguoi|"
    r"tep|dong|bang|hinh|neu|khi|voi|cho|mot|hai|va|la|co|nay|do|the|se|dang|"
    r"chi|tren|trong|tu|den|ra|vao|lam|thi|ma|nen|hay|cac|nhat|sau|truoc|con|"
    r"hon|sang|luu|xem|dat|goi|phai|tinh|dung|bo|thay|het|moi|"
    # Bổ sung sau đợt rà soát giao diện: tám chú thích trong apps/dashboard đi
    # qua bộ từ cũ vì không chứa từ nào trong danh sách trên. Ví dụ
    # "Nut dieu khien", "Chay suy luan Batch", "Tao index gia".
    r"giu|nguyen|nut|bam|chon|loc|tao|index|gia|chay|suy|luan|thu thap|"
    r"bat dau|dieu khien|lu|xac suat|phan phoi|thap|cao|luong|canh|bao|"
    r"danh|xuat|nhap|hien|noi|ket noi|diem)\b")

# Biểu tượng dạng emoji. Không tính ký tự vẽ khung U+2500..U+257F vì chúng là
# công cụ trình bày bảng trên terminal.
EMOJI = re.compile("[\U0001F000-\U0001FAFFℹ☀-➿️]")


def python_files():
    """Mọi tệp .py của dự án, bỏ qua thư mục đệm và chính tệp kiểm toán này.

    Tự loại mình ra vì các mẫu tìm kiếm bên dưới chứa đúng những ký tự mà chúng
    đi tìm, và các khối bắt lỗi ở đây là để bỏ qua tệp không phân tích được chứ
    không phải nuốt lỗi của dự án.
    """
    me = pathlib.Path(__file__).resolve()
    for directory in CODE_DIRS:
        base = PROJECT_ROOT / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" not in path.parts and path.resolve() != me:
                yield path


def notebooks():
    """Mọi notebook, kèm mã nguồn đã gỡ dòng lệnh magic."""
    for path in sorted((PROJECT_ROOT / "notebooks").rglob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        source = "\n".join(
            "" if line.lstrip().startswith(("!", "%")) else line
            for cell in cells if cell["cell_type"] == "code"
            for line in "".join(cell["source"]).split("\n"))
        yield path, cells, source


def text_files():
    """Tệp văn bản thường của dự án, nơi chú thích cũng phải viết có dấu.

    requirements.txt từng lọt qua năm vòng rà soát vì mọi phép kiểm chỉ nhìn
    .py và .ipynb. Nó là tệp người đọc mở đầu tiên khi dựng lại môi trường, và
    chú thích trong đó cũng là văn bản của dự án như mọi chỗ khác.
    """
    for name in ("requirements.txt", "pyproject.toml", ".gitignore"):
        path = PROJECT_ROOT / name
        if path.exists():
            yield path


def chuoi_tai_lieu(tree):
    """Mọi chuỗi tài liệu trong một cây cú pháp, kèm số dòng và tên chủ sở hữu."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                yield (getattr(node, "name", "<module>"),
                       getattr(node, "lineno", 1), doc)


def rel(path) -> str:
    """Đường dẫn tương đối so với gốc dự án, để đầu ra không lộ bố cục ổ đĩa."""
    return str(pathlib.Path(path).relative_to(PROJECT_ROOT))


# --- Từng phép kiểm ---------------------------------------------------------

def kiem_chu_thich_khong_dau():
    """Chú thích tiếng Việt viết không dấu.

    Trích dẫn tiếng Anh nguyên văn được bỏ qua: chúng không có từ tiếng Việt
    nào nên không lọt vào phép so từ khoá.
    """
    hits = []
    for path in python_files():
        lines = path.read_text(encoding="utf-8").split("\n")
        # Đánh dấu các dòng thuộc một trích dẫn nguyên văn tiếng Anh. Trích dẫn
        # mở bằng dấu nháy kép trong một dòng chú thích tiếng Việt và đóng bằng
        # dấu nháy kép ở dòng sau; các dòng ở giữa là văn bản gốc của bài báo,
        # sửa chúng là làm sai trích dẫn.
        in_quote = False
        quoted = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#"):
                in_quote = False
                continue
            n_quotes = stripped.count('"')
            if in_quote:
                quoted.add(i)
            if n_quotes % 2 == 1:
                in_quote = not in_quote
                if in_quote:
                    quoted.add(i)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#") or i in quoted:
                continue
            text = stripped[1:]
            if not text.strip() or any(c.lower() in VIET_CHARS for c in text):
                continue
            if VIET_WORDS.search(text.lower()):
                hits.append(f"{rel(path)}:{i + 1}  {stripped[:66]}")

    # Chuỗi tài liệu. Phép kiểm ở trên chỉ nhìn dòng bắt đầu bằng dấu thăng,
    # nên hai mươi chín chuỗi tài liệu viết không dấu trong dashboard và trong
    # scripts đã đi qua nó mà không bị chặn.
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for ten, lineno, doc in chuoi_tai_lieu(tree):
            # Xét TỪNG DÒNG chứ không xét cả khối. Một chuỗi tài liệu có dòng
            # đầu viết có dấu và các dòng sau viết không dấu sẽ lọt qua phép
            # kiểm xét cả khối, vì khối đó có chứa ký tự có dấu.
            # Xét TỪNG ĐOẠN chứ không xét cả khối, vì một chuỗi tài liệu
            # có dòng tóm tắt viết có dấu và phần thân viết không dấu sẽ lọt
            # qua phép kiểm xét cả khối. Xét theo đoạn chứ không theo dòng, vì
            # một câu có dấu bị ngắt dòng để lại dòng nối không mang dấu nào.
            for doan in re.split(r"\n\s*\n", doc):
                t = " ".join(doan.split())
                if not t:
                    continue
                # Dòng lệnh ví dụ không phải văn xuôi tiếng Việt. Bỏ qua,
                # nếu không thì "--ra reports/..." bị bắt vì chứa chữ "ra".
                if t.startswith(("python", "pip", "$env:", "PYTHONPATH", "-",
                                 "streamlit", "curl", "git")):
                    continue
                if any(c.lower() in VIET_CHARS for c in t):
                    continue
                if VIET_WORDS.search(t.lower()):
                    hits.append(f"{rel(path)}:{lineno}  chuỗi tài liệu của "
                                f"{ten}: {t[:56]}")

    # Tệp văn bản thường.
    for path in text_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            text = stripped[1:]
            if not text.strip() or any(c.lower() in VIET_CHARS for c in text):
                continue
            if VIET_WORDS.search(text.lower()):
                hits.append(f"{rel(path)}:{i}  {stripped[:66]}")
    return hits


def kiem_emoji():
    """Biểu tượng emoji trong mã, trái quy tắc trình bày đơn sắc."""
    hits = []
    for path in python_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for m in EMOJI.finditer(line):
                hits.append(f"{rel(path)}:{i}  {m.group()!r} U+{ord(m.group()):04X}")
    return hits


def kiem_duong_dan_ghi_cung():
    """Đường dẫn của một hệ điều hành ghi cứng, kể cả nằm trong chuỗi ký tự."""
    pattern = re.compile(r"[Cc]:[\\/]|/Users/[a-z]|/home/[a-z]|python\.exe|"
                         r"Python3[0-9]{2}")
    hits = []
    for path in python_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if pattern.search(line) and "Program Files" not in line \
                    and "Microsoft Word.app" not in line:
                hits.append(f"{rel(path)}:{i}  {line.strip()[:66]}")
    for path, _cells, source in notebooks():
        for i, line in enumerate(source.split("\n"), 1):
            if pattern.search(line):
                hits.append(f"{rel(path)} dòng {i}  {line.strip()[:60]}")
    return hits


def kiem_sinh_du_lieu_bia():
    """Dấu vết sinh dữ liệu không có thật trong tầng giao diện và tầng đo."""
    hits = []
    rnd = re.compile(r"np\.random|(?<![\w.])random\.")
    for path in python_files():
        if path.parts[path.parts.index(path.parts[0])] not in ("apps",):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if rnd.search(line) and not line.strip().startswith("#"):
                hits.append(f"{rel(path)}:{i}  sinh số ngẫu nhiên trong giao diện")
    # Giá trị mặc định trông như một phép đo
    metric_words = ("acc", "f1", "recall", "prec", "auc", "far", "score",
                    "rate", "latency", "loss", "metric", "conf")
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, (int, float))
                    and not isinstance(node.args[1].value, bool)):
                key = ast.unparse(node.args[0]).lower()
                if any(w in key for w in metric_words):
                    hits.append(f"{rel(path)}:{node.lineno}  "
                                f".get({ast.unparse(node.args[0])}, "
                                f"{node.args[1].value})")
    return hits


def kiem_nuot_loi():
    """Khối bắt ngoại lệ không báo ra ngoài."""
    hits = []
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = node.body
            if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Continue)):
                hits.append(f"{rel(path)}:{node.lineno}  except -> bỏ qua im lặng")
                continue
            # Trả về giá trị mà không in hay ném lại
            noisy = any(
                isinstance(n, ast.Raise)
                or (isinstance(n, ast.Call) and (
                    (isinstance(n.func, ast.Name) and n.func.id == "print")
                    or (isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("print", "error", "warning", "log",
                                            "warn", "exception"))))
                for stmt in body for n in ast.walk(stmt))
            # Chỉ tính là nuốt lỗi khi trả về một CON SỐ, vì con số trông y hệt
            # một phép đo thật. Trả None là cách báo "không tính được" mà mục
            # quy ước của dự án yêu cầu, nên không phải lỗi.
            returns_empty = any(
                isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, (int, float))
                and not isinstance(stmt.value.value, bool)
                for stmt in body)
            if returns_empty and not noisy:
                hits.append(f"{rel(path)}:{node.lineno}  except -> trả rỗng, không báo")
    return hits


def kiem_torch_load():
    """torch.load thiếu chế độ an toàn weights_only."""
    hits = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"):
                if not any(k.arg == "weights_only" for k in node.keywords):
                    hits.append(f"{rel(path)}:{node.lineno}  torch.load thiếu weights_only")
    return hits


def kiem_ma_chet():
    """Hàm, lớp, hằng số cấp module không nơi nào dùng tới.

    Đếm trên cây cú pháp, KHÔNG đếm tên xuất hiện trong chú thích: một tên chỉ
    được nhắc trong chú thích từng bị tính nhầm là đang dùng.
    """
    defined, used = {}, set()

    def thu_thap(tree, path):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith(("_", "test_", "Test")):
                    defined[node.name] = f"{rel(path)}:{node.lineno}"
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        defined[t.id] = f"{rel(path)}:{node.lineno}"

    def dung(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                used.update(re.findall(r"\w+", node.value))
            elif isinstance(node, ast.alias):
                used.add(node.name.split(".")[-1])
                if node.asname:
                    used.add(node.asname)

    trees = []
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        trees.append((tree, path))
        thu_thap(tree, path)
    for tree, _ in trees:
        dung(tree)
    for _path, _cells, source in notebooks():
        try:
            dung(ast.parse(source))
        except SyntaxError:
            pass

    return [f"{loc}  {name}" for name, loc in sorted(defined.items())
            if name not in used]


# Dòng đánh dấu mà một tệp tài liệu tự khai rằng nhiệm vụ của nó là kể lại
# việc đã xoá. Dùng dấu hiệu trong nội dung thay vì danh sách tên tệp, vì danh
# sách tên phình ra mỗi lần viết thêm một tệp ghi nhận, và người viết tệp mới
# không có cách nào biết rằng phải sửa bộ kiểm toán kèm theo.
MIEN_NHAC_TEN_DA_XOA = "kiem-toan: tep nay ghi nhan viec da xoa"


def _duoc_mien_nhac_ten_da_xoa(path) -> bool:
    """Tệp có được phép nhắc tên đã xoá không.

    Hai trường hợp. Một, tệp tự khai bằng dòng đánh dấu ở trên. Hai, đề cương
    chi tiết là tài liệu ĐÃ NỘP nên không được sửa; lệch giữa đề cương và sản
    phẩm là việc phải giải trình trong báo cáo, không phải lỗi tài liệu.
    """
    if path.name == "DE_CUONG_CHI_TIET.md":
        return True
    if path.suffix != ".md":
        return False
    return MIEN_NHAC_TEN_DA_XOA in path.read_text(encoding="utf-8")


def kiem_tham_chieu_da_xoa():
    """Tài liệu hoặc mã còn nhắc những thứ đã bị xoá khỏi dự án."""
    # Thu thập mọi tên đang thật sự tồn tại trong mã.
    ton_tai = set()
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        ton_tai.add(path.stem)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ton_tai.add(node.name)

    # Những tên từng tồn tại và đã bị xoá; nhắc lại chúng là tài liệu lỗi thời.
    da_xoa = ["GNNExplainer", "XAIExporter", "LinkPredictor", "TimeEncoder",
              "NFUNSWNB15DatasetLoader", "ProjectConfig", "dataset_loader",
              "generate_sample_data", "comparison_table",
              "UNCERTAIN_ENTROPY_THRESHOLD", "UNCERTAIN_MARGIN_CEILING"]
    hits = []
    # Quét cả chú thích trong mã và tệp văn bản thường, không riêng .md.
    # requirements.txt nhắc `dataset_loader` suốt nhiều vòng rà soát vì phép
    # kiểm này chỉ nhìn thư mục tài liệu.
    docs = [p for p in PROJECT_ROOT.rglob("*.md")
            if "BACKUP" not in str(p) and ".git" not in p.parts]
    docs += list(text_files()) + list(python_files())
    for name in da_xoa:
        if name in ton_tai:
            continue
        for path in docs:
            if _duoc_mien_nhac_ten_da_xoa(path):
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                if name in line:
                    hits.append(f"{rel(path)}:{i}  còn nhắc `{name}` đã xoá")

    # Tên đã xoá còn sót trong ô markdown của notebook. Ô mã đã được phép kiểm
    # mã chết lo, còn phần trình bày thì chưa ai nhìn tới.
    for path, cells, _source in notebooks():
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "markdown":
                continue
            noi_dung = "".join(cell["source"])
            for name in da_xoa:
                if name not in ton_tai and name in noi_dung:
                    hits.append(f"{rel(path)} ô {i}  còn nhắc `{name}` đã xoá")
    return hits


def kiem_notebook():
    """Notebook hỏng cú pháp hoặc lộ đường dẫn máy trong đầu ra."""
    hits = []
    for path, cells, _source in notebooks():
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            src = "\n".join("" if l.lstrip().startswith(("!", "%")) else l
                            for l in "".join(cell["source"]).split("\n"))
            try:
                ast.parse(src)
            except SyntaxError as error:
                hits.append(f"{rel(path)} ô {i}  lỗi cú pháp: {error}")
            for out in cell.get("outputs", []):
                if "/Users/" in str(out) or re.search(r"[Cc]:\\\\Users", str(out)):
                    hits.append(f"{rel(path)} ô {i}  đầu ra lộ đường dẫn máy")
                    break
        # Từ nbformat 4.5 mỗi ô bắt buộc có trường id. Ô thêm tay mà quên
        # trường này vẫn mở được trong Jupyter nên lỗi không lộ ra khi dùng,
        # chỉ lộ khi chạy nbformat.validate.
        for i, cell in enumerate(cells):
            if "id" not in cell:
                hits.append(f"{rel(path)} ô {i}  thiếu trường id")
    return hits


def kiem_ky_tu_an():
    """Ký tự Unicode ẩn trong tài liệu và mã."""
    hits = []
    targets = list(python_files()) + [
        p for p in PROJECT_ROOT.rglob("*.md")
        if "BACKUP" not in str(p) and ".git" not in p.parts]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for ch in set(text):
            if unicodedata.category(ch) in ("Cf", "Co", "Cs") and ch != "﻿":
                hits.append(f"{rel(path)}  ký tự ẩn U+{ord(ch):04X}")
    return hits


# Tên viết tắt trong sơ đồ kiến trúc, ứng với tên tệp mã.
SO_DO_NHAN = {
    # Tên tệp mã.
    "preprocess": "PREP", "graph_builder": "GB", "labels": "LBL",
    "protocols": "PROTO", "tgn_memory": "MEM", "embedding": "EMB",
    "edge_decoder": "DEC", "anomaly_detector": "DET", "runner": "RUN",
    "ood_eval": "OOD", "paper_metrics": "PM", "figures": "FIG",
    "format": "FMT", "gnn_explainer": "TGNX", "subgraph_extractor": "SUB",
    # Tên lớp, vì nhiều nơi nhập theo dạng `from tgn_nids.explainers import X`
    # nên tên module không xuất hiện trong câu lệnh.
    "TGNExplainer": "TGNX", "GNNExplainer": "GNNX",
    "SubgraphExtractor": "SUB", "NetFlowPreprocessor": "PREP",
    "TemporalGraphBuilder": "GB",
}


def kiem_so_do_kien_truc():
    """Đối chiếu từng mũi tên của sơ đồ kiến trúc với câu lệnh import thật.

    Một sơ đồ vẽ sai còn tệ hơn không có sơ đồ: người đọc tin nó và đi tìm quan
    hệ không tồn tại. Phép kiểm này chạy hai chiều, bắt cả mũi tên thừa lẫn
    liên kết bị thiếu.
    """
    def targets(src):
        out = set()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return out
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[-1] in SO_DO_NHAN:
                    out.add(SO_DO_NHAN[node.module.split(".")[-1]])
                for alias in node.names:      # from tgn_nids.utils import figures
                    if alias.name in SO_DO_NHAN:
                        out.add(SO_DO_NHAN[alias.name])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[-1] in SO_DO_NHAN:
                        out.add(SO_DO_NHAN[alias.name.split(".")[-1]])
        return out

    thuc_te = set()
    for path in python_files():
        if path.stem not in SO_DO_NHAN:
            continue
        for tgt in targets(path.read_text(encoding="utf-8")):
            thuc_te.add((SO_DO_NHAN[path.stem], tgt))
    ten_nb = {"NB1_train_v2": "NB1", "NB2_train_v3": "NB2",
              "NB3_phan_tich": "NB3"}
    for path, _cells, source in notebooks():
        if path.stem in ten_nb:
            for tgt in targets(source):
                thuc_te.add((ten_nb[path.stem], tgt))

    so_do = PROJECT_ROOT / "docs" / "SO_DO_KIEN_TRUC_DU_AN.md"
    if not so_do.exists():
        return ["thiếu docs/SO_DO_KIEN_TRUC_DU_AN.md"]
    doc = so_do.read_text(encoding="utf-8")
    start = doc.index("flowchart TD")
    block = doc[start:doc.index("```", start)]

    ve = set()
    for line in block.split("\n"):
        m = re.match(r"\s*([A-Z0-9&\s]+?)\s*-->\s*([A-Z0-9&\s]+?)\s*$", line)
        if not m:
            continue
        for a in m.group(1).split("&"):
            for b in m.group(2).split("&"):
                ve.add((a.strip(), b.strip()))

    quan_tam = set(SO_DO_NHAN.values()) | {"NB1", "NB2", "NB3"}
    ve = {x for x in ve if x[0] in quan_tam and x[1] in quan_tam}
    thuc_te = {x for x in thuc_te if x[0] in quan_tam and x[1] in quan_tam}

    hits = [f"sơ đồ vẽ {a} -> {b} mà mã không có" for a, b in sorted(ve - thuc_te)]
    hits += [f"mã có {a} -> {b} mà sơ đồ thiếu" for a, b in sorted(thuc_te - ve)]
    return hits


def kiem_so_do_liet_ke_du_tep():
    """Sơ đồ khối công cụ và khối kiểm thử phải liệt kê đúng số tệp đang có.

    Phép kiểm sơ đồ kiến trúc phía trên chỉ đối chiếu khối chính. Hai khối còn
    lại là danh sách phẳng, nên chúng lệch thầm lặng mỗi khi một tệp bị xoá
    hoặc thêm mà không ai sửa sơ đồ.
    """
    so_do = PROJECT_ROOT / "docs" / "SO_DO_KIEN_TRUC_DU_AN.md"
    if not so_do.exists():
        return ["thiếu docs/SO_DO_KIEN_TRUC_DU_AN.md"]
    noi_dung = so_do.read_text(encoding="utf-8")

    hits = []
    for thu_muc, duoi in (("scripts", "*.py"), ("tests", "test_*.py")):
        base = PROJECT_ROOT / thu_muc
        if not base.exists():
            continue
        for path in sorted(base.rglob(duoi)):
            if "__pycache__" in path.parts:
                continue
            if path.name not in noi_dung:
                hits.append(f"{rel(path)}  có thật nhưng thiếu trong sơ đồ")

    # Chiều ngược lại: sơ đồ nhắc tệp .py nào không còn tồn tại.
    for m in re.finditer(r"([A-Za-z0-9_]+\.py)", noi_dung):
        ten = m.group(1)
        co = any((PROJECT_ROOT / d).rglob(ten) and
                 list((PROJECT_ROOT / d).rglob(ten))
                 for d in ("scripts", "tests", "src", "apps"))
        if not co:
            hits.append(f"docs/SO_DO_KIEN_TRUC_DU_AN.md  nhắc {ten} "
                        f"nhưng tệp không còn")
    return hits


def kiem_bang_so_lieu_sinh_tu_dong():
    """Tài liệu số liệu sinh tự động đã cũ hơn kho kết quả.

    docs/KET_QUA_DO_LUONG.md là bản duy nhất trong docs/ không viết tay, nên nó
    là nơi an toàn để trích số. An toàn đó chỉ đúng khi nó được dựng lại sau
    lần chạy gần nhất.
    """
    doc = PROJECT_ROOT / "docs" / "KET_QUA_DO_LUONG.md"
    if not doc.exists():
        return ["thiếu docs/KET_QUA_DO_LUONG.md, "
                "chạy python scripts/tong_hop_ket_qua.py"]
    ket_qua = sorted((PROJECT_ROOT / "models" / "saved").rglob("result_*.json"))
    if not ket_qua:
        return []
    moi_nhat = max(p.stat().st_mtime for p in ket_qua)
    if doc.stat().st_mtime < moi_nhat:
        return ["docs/KET_QUA_DO_LUONG.md cũ hơn tệp kết quả mới nhất, "
                "chạy lại python scripts/tong_hop_ket_qua.py"]
    return []


CHECKS = [
    ("Chú thích tiếng Việt không dấu", kiem_chu_thich_khong_dau),
    ("Emoji trong mã", kiem_emoji),
    ("Đường dẫn máy ghi cứng", kiem_duong_dan_ghi_cung),
    ("Dấu vết sinh dữ liệu không thật", kiem_sinh_du_lieu_bia),
    ("Khối bắt lỗi nuốt ngoại lệ", kiem_nuot_loi),
    ("torch.load thiếu chế độ an toàn", kiem_torch_load),
    ("Mã chết", kiem_ma_chet),
    ("Tài liệu nhắc thứ đã xoá", kiem_tham_chieu_da_xoa),
    ("Notebook", kiem_notebook),
    ("Ký tự Unicode ẩn", kiem_ky_tu_an),
    ("Sơ đồ kiến trúc khớp mã", kiem_so_do_kien_truc),
    ("Sơ đồ liệt kê đủ tệp", kiem_so_do_liet_ke_du_tep),
    ("Bảng số liệu sinh tự động còn mới", kiem_bang_so_lieu_sinh_tu_dong),
]


def main() -> int:
    """Chạy toàn bộ phép kiểm tĩnh và trả mã lỗi nếu còn bất cập."""
    tong = 0
    for ten, ham in CHECKS:
        hits = ham()
        tong += len(hits)
        trang_thai = "đạt" if not hits else f"{len(hits)} phát hiện"
        print(f"{ten:.<46}{trang_thai}")
        for h in hits:
            print(f"    {h}")
    print()
    print(f"TỔNG: {tong} phát hiện")
    return 1 if tong else 0


if __name__ == "__main__":
    sys.exit(main())
