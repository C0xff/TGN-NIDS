<!-- kiem-toan: tep nay ghi nhan viec da xoa, nen duoc phep nhac ten cac thu da bi xoa khoi du an -->

# Sơ đồ kiến trúc và móc nối giữa các tệp

Mọi mũi tên trong tài liệu này đều
dựng từ câu lệnh `import` thật trong mã, không vẽ theo trí nhớ.

---

## 1. Khối chính: đường đi từ dữ liệu thô tới kết quả

Đây là khối làm nên đồ án. Đọc từ trên xuống là đúng thứ tự dữ liệu chảy qua.

```mermaid
flowchart TD
    RAW[("data/raw/<br/>6 tệp Parquet")]

    subgraph NB["notebooks/twoDTS - nơi điều phối"]
        NB1["NB1_train_v2.ipynb<br/>huấn luyện trên v2"]
        NB2["NB2_train_v3.ipynb<br/>huấn luyện trên v3"]
        NB3["NB3_phan_tich.ipynb<br/>phân tích, giải thích"]
    end

    subgraph DATA["src/tgn_nids/data - chuẩn bị dữ liệu"]
        PROTO["protocols.py<br/>chia tập theo giao thức"]
        PREP["preprocess.py<br/>chuẩn hoá thang đo"]
        LBL["labels.py<br/>quy đổi nhãn"]
        GB["graph_builder.py<br/>dựng đồ thị thời gian"]
    end

    subgraph MODEL["src/tgn_nids/models - ba khối mạng"]
        MEM["tgn_memory.py<br/>bộ nhớ theo đỉnh"]
        EMB["embedding.py<br/>attention trên đồ thị"]
        DEC["edge_decoder.py<br/>đầu giai đoạn 1"]
        DET["anomaly_detector.py<br/>đầu giai đoạn 2"]
    end

    subgraph EXP["src/tgn_nids/experiments - điều khiển"]
        RUN["runner.py<br/>vòng huấn luyện"]
        OOD["ood_eval.py<br/>đánh giá out-of-distribution"]
    end

    subgraph UTIL["src/tgn_nids/utils - đo và vẽ"]
        PM["paper_metrics.py<br/>tính mọi chỉ số"]
        FIG["figures.py<br/>vẽ hình báo cáo"]
        FMT["format.py<br/>làm tròn không nói dối"]
    end

    subgraph XAI["src/tgn_nids/explainers - giải thích"]
        SUB["subgraph_extractor.py<br/>subgraph k-hop"]
        GNNX["gnn_explainer.py<br/>GNNExplainer: trọng số cạnh"]
    end

    OUT[("models/saved/<br/>trọng số + result_*.json")]

    RAW --> NB1 & NB2 & NB3
    NB1 & NB2 --> RUN
    NB1 & NB2 & NB3 --> PREP
    NB3 --> OOD
    NB3 --> SUB
    NB3 --> GNNX

    RUN --> PROTO & GB
    RUN --> MEM & EMB & DEC & DET
    RUN --> PM
    OOD --> PREP & PM
    OOD --> MEM & EMB & DET
    GB --> LBL

    NB1 & NB2 & NB3 --> FIG
    FIG --> FMT

    RUN --> OUT
    OUT --> NB3
```

**Cách đọc.** Notebook là nơi điều phối, không phải nơi chứa thuật toán: chúng
nạp dữ liệu thô, dựng `NetFlowPreprocessor`, gọi `runner.py`, rồi vẽ hình. Toàn
bộ phần tính toán nằm trong `src/tgn_nids/`. Nhờ tách như vậy, đổi một siêu tham
số chỉ phải sửa ô tham số của notebook, không đụng tới mã mô hình.

Chú ý chiều mũi tên ở `preprocess.py`: **notebook tạo ra bộ tiền xử lý rồi truyền
vào `runner`**, chứ `runner` không tự nhập nó. Nhờ vậy cùng một bộ tiền xử lý đã
khớp trên tập huấn luyện được dùng lại nguyên vẹn cho tập kiểm thử và cho
dashboard, thay vì mỗi nơi khớp một bộ riêng.

`labels.py` chỉ có một nơi gọi là `graph_builder.py`, tại bước quy đổi nhãn lúc
dựng đồ thị.

Ba công cụ giải thích trả lời ba câu hỏi khác nhau và nằm ở ba nơi khác nhau.
`SubgraphExtractor` cho biết cảnh báo nằm ở đâu trong mạng, notebook gọi trực
tiếp. `GNNExplainer` cho biết cạnh lân cận nào giữ quyết định, cũng do notebook
gọi. `TGNExplainer` cho biết đặc trưng nào đẩy quyết định, và **chỉ dashboard
gọi**; notebook tự triệt tiêu từng cột ngay trong ô mã rồi gọi lại
`evaluate_ood_binary`, nên không đi qua lớp đó.

Cả ba đều **hậu nghiệm**: chúng nạp mô hình đã lưu, đóng băng trọng số, và
`runner.py` không nhập tệp nào trong nhóm này. Ô mã của mục 8 trong notebook còn
so trọng số trước và sau mỗi lần giải thích, dừng với lỗi nếu có tham số nào bị
đổi.

**Ba khối mạng nối tiếp nhau** theo đúng thứ tự `tgn_memory` rồi `embedding` rồi
một trong hai đầu ra. Giai đoạn một dùng `edge_decoder`, giai đoạn hai dùng
`anomaly_detector`. Đây là lý do checkpoint của hai giai đoạn có tiền tố khoá
khác nhau, `decoder.` và `detector.`.

---

## 2. Khối trình diễn: dashboard

Khối này đọc kết quả của khối chính, không sinh ra kết quả mới.

```mermaid
flowchart TD
    SAVED[("models/saved/<br/>trọng số + kết quả")]
    DEMO[("data/samples/<br/>3 tệp demo")]

    APP["app.py<br/>giao diện Streamlit"]
    IE["inference_engine.py<br/>nạp mô hình, suy luận"]
    VIZ["visualization.py<br/>biểu đồ Plotly"]
    THEME["theme.py<br/>bảng màu, CSS"]

    MEM2["models/tgn_memory.py"]
    EMB2["models/embedding.py"]
    DET2["models/anomaly_detector.py"]
    PREP2["data/preprocess.py"]
    TGNX2["explainers/gnn_explainer.py"]

    FMT2["utils/format.py"]

    APP --> IE & VIZ & THEME
    SAVED --> IE
    DEMO --> APP
    IE --> MEM2 & EMB2 & DET2 & PREP2 & TGNX2
    IE --> FMT2
```

**Điểm cần nhớ.** Dashboard dùng lại đúng ba khối mạng của khối chính, nên nó
không thể "chạy khác" mô hình đã huấn luyện. Nó cũng dùng chung `utils/format.py`
với phần báo cáo, nên một tỷ lệ hiện trên màn hình và cùng tỷ lệ đó in trong báo
cáo đi qua đúng một phép làm tròn.

Chỗ còn cài hai bản: dashboard **tự dựng đồ thị lân cận** bằng mã riêng trong
`inference_engine.py` chứ không gọi `subgraph_extractor.py`. Hai bản từng khác
nhau về chiều duyệt, và lỗi chỉ lộ ra khi lần đầu có người gọi bản thứ hai.

---

## 3. Khối công cụ: scripts

Không tệp nào ở đây tham gia vào việc sinh ra con số của báo cáo. Chúng là công
cụ vận hành.

```mermaid
flowchart LR
    subgraph DATA_TOOLS["Chuẩn bị dữ liệu (scripts/data/)"]
        DL["download_nids_datasets.py<br/>tải bộ dữ liệu"]
        RB["rebuild_parquet.py<br/>dựng lại Parquet từ CSV"]
        PROF["profile_datasets.py<br/>bảng quy mô dữ liệu"]
        BDS["build_demo_samples.py<br/>sinh 3 tệp demo"]
    end

    subgraph KAGGLE["Vòng chạy Kaggle (scripts/ops/)"]
        PKG["package_for_kaggle.py<br/>đóng gói ZIP"]
        ACC["kaggle_account.py<br/>đổi tài khoản"]
        COL["collect_kaggle_results.py<br/>tải kết quả về"]
    end

    subgraph CHECK["Kiểm tra trước khi chạy (scripts/pipeline/ & scripts/ops/)"]
        CN["check_notebooks.py"]
        CT["chay_thu_notebook.py"]
        KT["kiem_toan_du_an.py<br/>kiểm toán toàn dự án"]
        KTH["kiem_tep_hong.py<br/>kiểm tệp nén và checkpoint còn đọc được"]
    end

    subgraph MISC["Vận hành & Đồng bộ (scripts/ops/)"]
        BK["backup_ban_hoan_thien.py<br/>sao lưu ra ngoài dự án"]
    end

    subgraph ANALYSIS["Tổng hợp số liệu (scripts/analysis/)"]
        THKQ["tong_hop_ket_qua.py<br/>sinh docs/KET_QUA_DO_LUONG.md"]
        SSLDR["sinh_so_lieu_dau_ra.py<br/>sinh docs/SO_LIEU_DAU_RA_CHI_TIET.md"]
        PROF["profile_datasets.py<br/>khảo sát bộ dữ liệu thô"]
        DTD["do_tap_demo.py<br/>đo ba tệp demo, ghi ra models/saved/twoDTS_demo"]
    end
```

Các công cụ vận hành phục vụ tự động hóa chu trình chuẩn bị dữ liệu, điều phối huấn luyện từ xa trên Kaggle, và thẩm định tính toàn vẹn của dự án trước và sau mỗi phiên làm việc. Chúng độc lập với mã nguồn lõi của mô hình và không can thiệp vào các tham số đã khóa.

---

## 4. Khối kiểm thử

```mermaid
flowchart LR
    T1["test_data_pipeline.py"] --> GB2["data/graph_builder.py"]
    T1 --> SIM["simulation/generator.py"]
    T2["test_edge_decoder.py"] --> DEC3["models/edge_decoder.py"]
    T3["test_graph_builder_labels.py"] --> GB2
    T3 --> LBL3["data/labels.py"]
    T4["test_paper_metrics.py"] --> PM3["utils/paper_metrics.py"]
    T5["test_protocols.py"] --> PROTO3["data/protocols.py"]
    T6["test_runner_guards.py"] --> RUN3["experiments/runner.py"]
    T6 --> PREP3["data/preprocess.py"]
    T7["test_build_demo_samples.py"] --> BDS["scripts/pipeline/build_demo_samples.py"]
```

Bảy tệp kiểm thử, 65 phép. Chúng phủ phần dữ liệu và phần đo, không phủ phần
giao diện: dashboard kiểm bằng cách chạy thật với ba tệp demo.

---

## 5. Những tệp nằm ngoài mọi sơ đồ trên

Ba nhóm dưới đây không có mũi tên nào nối tới mã đang chạy.

**Notebook đã thực thi** trong `notebooks/executed/`: ba tệp `.ipynb` giữ nguyên
đầu ra của một lần chạy cụ thể, chỉ để đọc và làm bằng chứng. Không tệp nào nạp
chúng.

**Tài liệu và kết quả**: toàn bộ `docs/`, `reports/`, `models/saved/`,
`data/` là dữ liệu chứ không phải mã. `models/saved/` là điểm nối duy nhất giữa
khối chính và khối trình diễn.

---

## 6. Ba quy tắc rút ra từ sơ đồ

**Một, mọi con số đi qua đúng một cửa.** `paper_metrics.py` là nơi duy nhất tính
chỉ số, cho cả huấn luyện lẫn đánh giá out-of-distribution. Sửa sai ở đó thì sai
toàn bộ báo cáo, nên đó là tệp cần đọc kỹ nhất khi nghi ngờ kết quả.

**Hai, dashboard không có đường tắt.** Nó nạp đúng ba khối mạng của khối chính.
Nếu màn hình hiện số khác báo cáo thì nguyên nhân nằm ở dữ liệu đầu vào hoặc
điểm làm việc, không phải ở chỗ mô hình khác nhau.

**Ba, một chức năng cài hai nơi là mầm lỗi.** Việc dựng đồ thị lân cận tồn tại
cả trong `subgraph_extractor.py` lẫn trong `inference_engine.py`. Hai bản từng
khác nhau về chiều duyệt, và lỗi chỉ lộ ra khi lần đầu có người gọi bản thứ
nhất. Chỗ nào còn trùng lặp như vậy thì phải hoặc gộp lại, hoặc ghi rõ trong mã
rằng có hai bản và vì sao.
