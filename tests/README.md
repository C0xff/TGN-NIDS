# Thư mục `tests`

Đây là nơi đặt các kiểm thử của dự án. Mỗi tệp tập trung vào một nhóm hành vi
cần được giữ ổn định: dữ liệu đầu vào, cách chia tập, chỉ số đánh giá, mô hình
và các ràng buộc bảo vệ trong quá trình thí nghiệm.

## Các tệp kiểm thử

| Tệp | Chức năng |
| --- | --- |
| `__init__.py` | Đánh dấu `tests` là một gói Python. |
| `test_build_demo_samples.py` | Kiểm tra lấy mẫu phân tầng cho dữ liệu minh họa: đúng số lượng, đủ lớp và đúng thứ tự thời gian. |
| `test_data_pipeline.py` | Kiểm tra toàn tuyến xử lý dữ liệu mẫu, từ đọc CSV đến tạo đồ thị thời gian `TemporalData`. |
| `test_edge_decoder.py` | Kiểm tra cơ chế che thuộc tính, tái thiết đặc trưng cạnh, điểm bất thường và đầu phân loại cạnh. |
| `test_graph_builder_labels.py` | Kiểm tra nhãn nhị phân, nhãn đa lớp và bảo đảm nhãn không lọt vào đặc trưng cạnh. |
| `test_paper_metrics.py` | Kiểm tra các chỉ số nhị phân, đa lớp và tên gọi chỉ số dùng khi đối chiếu bài báo. |
| `test_protocols.py` | Kiểm tra các giao thức TE-G-SAGE, GraphIDS và Anomal-E: tỷ lệ chia, phân tầng, thứ tự thời gian và tính tái lập. |
| `test_runner_guards.py` | Kiểm tra các ràng buộc cấu hình, trạng thái tiền xử lý, ablation và tệp đầu ra của một lần chạy. |

## Quy ước chung

- Tên kiểm thử mô tả hành vi cần bảo vệ.
- Dữ liệu kiểm thử nhỏ, cố định hạt giống và không phụ thuộc GPU.
- Mỗi kiểm thử chỉ kiểm tra một điều kiện chính để lỗi trả về dễ đọc.
- Trường hợp hồi quy phải tái hiện đúng điều kiện có thể làm sai dữ liệu, nhãn hoặc chỉ số.
