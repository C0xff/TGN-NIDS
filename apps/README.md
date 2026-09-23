# Thư mục `apps`

Thư mục này chứa ứng dụng Streamlit của đồ án. Dashboard nhận dữ liệu NetFlow,
gọi mô hình đã lưu để suy luận, trình bày chỉ số và hiển thị phần giải thích cho
các cảnh báo mạng.

## Cấu trúc thư mục

```text
apps/
├── README.md
└── dashboard/
    ├── __init__.py
    ├── app.py
    ├── inference_engine.py
    ├── visualization.py
    ├── theme.py
    ├── style.css
    └── static/vendor/
        ├── bindings/utils.js
        ├── tom-select/tom-select.css
        ├── tom-select/tom-select.complete.min.js
        └── vis-9.1.2/
            ├── vis-network.css
            └── vis-network.min.js
```

## Các tệp của dashboard

| Tệp | Chức năng |
| --- | --- |
| `dashboard/__init__.py` | Đánh dấu gói giao diện dashboard. |
| `dashboard/app.py` | Điều phối giao diện Streamlit, trạng thái phiên, nguồn dữ liệu và các tab chức năng. |
| `dashboard/inference_engine.py` | Nạp checkpoint, chuẩn bị dữ liệu, chạy suy luận, quản lý bộ nhớ TGN và trả kết quả cho giao diện. |
| `dashboard/visualization.py` | Tạo biểu đồ Plotly, bảng HTML, đồ thị lân cận và phần hiển thị cảnh báo. |
| `dashboard/theme.py` | Khai báo token màu sáng/tối và sinh CSS ghi đè theo chủ đề đang chọn. |
| `dashboard/style.css` | Định kiểu các khối giao diện Streamlit và HTML tùy biến. |

## `static/vendor/`

Các tệp trong `static/vendor/` là thư viện bên thứ ba được nhúng sẵn để phần
đồ thị và ô chọn của dashboard hoạt động không cần tải tài nguyên từ mạng.

| Tệp | Chức năng |
| --- | --- |
| `bindings/utils.js` | Mã kết nối giữa Pyvis và các thành phần JavaScript trên trang. |
| `tom-select/tom-select.css` | Kiểu hiển thị cho ô chọn có khả năng tìm kiếm. |
| `tom-select/tom-select.complete.min.js` | Thư viện Tom Select đã nén. |
| `vis-9.1.2/vis-network.css` | Kiểu hiển thị của đồ thị mạng Vis Network. |
| `vis-9.1.2/vis-network.min.js` | Thư viện Vis Network đã nén. |

Các bundle trong `static/vendor/` không thuộc mã nguồn do đồ án tự phát triển;
không chỉnh sửa comment, mã nén hoặc định dạng của chúng khi bảo trì dashboard.
