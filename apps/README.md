# Thư mục `apps`

Thư mục này chứa ứng dụng Streamlit của đồ án. Dashboard nhận tệp NetFlow,
gọi mô hình đã lưu để suy luận, trình bày chỉ số và hiển thị phần giải thích cho
các cảnh báo mạng. Đây là công cụ minh họa và kiểm thử chức năng trong môi
trường ngoại tuyến, chưa phải hệ thống giám sát lưu lượng trực tiếp.

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

Giao diện cho phép khôi phục bộ nhớ TGN từ checkpoint hoặc đặt bộ nhớ về 0
trước mỗi lượt phân tích. Hai lựa chọn có thể cho kết quả khác nhau khi dữ liệu
có địa chỉ mới. Đồ thị lân cận và độ quan trọng thuộc tính trong thẻ XAI được
tính trên dữ liệu đang xem. Kết quả GNNExplainer của báo cáo được tạo ngoại
tuyến và không phải một chức năng của dashboard.

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

## Tài sản ngoài GitHub

Kho công khai không kèm checkpoint hoặc dữ liệu mẫu. Khi thiếu các tệp này,
dashboard chỉ hoạt động ở chế độ trình diễn để kiểm tra luồng xử lý và giao
diện. Gói Google Drive đi kèm lưu các tài sản cần thiết khi muốn chạy suy luận
bằng trọng số đã chốt.

Ba tệp demo trong gói nội bộ được lấy mẫu để kiểm tra giao diện. Trường hợp
thiếu địa chỉ IP dùng định danh tổng hợp để phần mềm tiếp tục chạy. Các chỉ số
từ trường hợp này không đại diện cho hiệu năng khoa học của mô hình đồ thị.
