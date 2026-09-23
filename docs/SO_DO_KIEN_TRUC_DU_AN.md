# Sơ đồ kiến trúc dự án và vị trí tài sản

Tài liệu này mô tả toàn bộ đồ án theo hai nơi lưu trữ: mã và tài liệu kỹ thuật
công khai trên GitHub; dữ liệu, đầu ra thực nghiệm và tài liệu toàn văn trong
gói Google Drive đi kèm. Sơ đồ không ngụ ý rằng các tài sản trên Drive có mặt
trong bản sao GitHub.

## 1. Phần công khai trên GitHub

```mermaid
flowchart TD
    SRC["src/tgn_nids/<br/>tiền xử lý, đồ thị, TGN, đánh giá, giải thích"]
    APP["apps/dashboard/<br/>giao diện Streamlit"]
    SCRIPTS["scripts/<br/>chuẩn bị, kiểm tra, tổng hợp"]
    TESTS["tests/<br/>kiểm thử tự động"]
    EXEC["notebooks/executed/<br/>bốn notebook đã thực thi"]
    DOCS["docs/<br/>tài liệu kỹ thuật và số liệu đối chiếu"]

    SCRIPTS --> SRC
    APP --> SRC
    TESTS --> SRC
    EXEC --> SRC
    EXEC --> DOCS
    SCRIPTS --> DOCS
```

`src/tgn_nids/` là thư viện dùng chung. Dashboard, script, kiểm thử và notebook
đã thực thi đều dựa vào các thành phần trong thư viện này; `docs/` giải thích
kiến trúc, giao thức đo và kết quả đã công bố.

## 2. Tài sản trong gói Google Drive

```mermaid
flowchart LR
    subgraph DRIVE["Google Drive — tài sản không công bố trên GitHub"]
        DATA["data/<br/>NetFlow gốc, CSV/Parquet, tệp mẫu"]
        SAVED["models/saved/<br/>checkpoint, chỉ số, bảng, hình"]
        DIST["dist/<br/>gói chuyển giao Kaggle"]
        REFS["references/<br/>toàn văn, RFC, bản trích xuất"]
        REPORTS["reports/<br/>báo cáo Word và hình gốc"]
    end

    DATA --> SRC["Mã nguồn GitHub"]
    SAVED --> APP["Dashboard GitHub"]
    SAVED --> DOCS["Tài liệu GitHub"]
    REFS --> DOCS
    DIST --> SCRIPTS["Script GitHub"]
    SAVED --> REPORTS
    DATA --> REPORTS
    REFS --> REPORTS
```

Các mũi tên biểu diễn quan hệ sử dụng hoặc đối chiếu, không phải cơ chế tự động
tải tệp. Khi clone kho GitHub, các thư mục trên Drive không xuất hiện trong cây
làm việc.

## 3. Luồng sử dụng

1. Cài môi trường từ `requirements.txt` và mã trong `src/`.
2. Chạy `pytest` để kiểm tra phần công khai không cần dữ liệu lớn.
3. Dùng dashboard ở chế độ trình diễn nếu chưa có checkpoint.
4. Khi cần tái tạo môi trường dữ liệu, suy luận bằng trọng số đã chốt hoặc đối
   chiếu số liệu gốc, lấy đúng thư mục tương ứng từ gói Drive.
5. Dùng danh mục trong `references/TAI_LIEU_THAM_KHAO.md` trên GitHub để tra
   cứu; chỉ xem bản toàn văn trong Drive khi có quyền sử dụng phù hợp.

## 4. Phân định trách nhiệm

| Nơi lưu | Vai trò |
| --- | --- |
| GitHub | Mã nguồn, kiểm thử, dashboard, script, tài liệu kỹ thuật và notebook đã thực thi. |
| Google Drive | Dữ liệu lớn, checkpoint, kết quả đã chốt, gói bàn giao, báo cáo hoàn chỉnh và toàn văn tài liệu tham khảo. |

Không ghi đè dữ liệu, checkpoint hoặc đầu ra đã chốt trong gói Drive khi chỉ
cần kiểm tra hoặc trình bày lại kết quả.
