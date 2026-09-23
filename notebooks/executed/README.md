# Notebook đã thực thi

Thư mục này lưu bốn notebook đã chạy hoàn tất, kèm đầu ra của từng ô. Đây là
nguồn đối chiếu trực tiếp cho quy trình huấn luyện, phân tích và thí nghiệm cắt
bỏ được trình bày trong đồ án.

## Cấu trúc

```text
executed/
├── README.md
├── b-twodts-train-v2.ipynb
├── b-twodts-train-v3.ipynb
├── twodts-phan-tich.ipynb
└── b-twodts-ablation.ipynb
```

## Nội dung từng notebook

| Tệp | Nội dung |
| --- | --- |
| `b-twodts-train-v2.ipynb` | Huấn luyện hai giai đoạn và đánh giá trên NF-UNSW-NB15-v2. |
| `b-twodts-train-v3.ipynb` | Huấn luyện trên NF-UNSW-NB15-v3 và NF-CSE-CIC-IDS2018-v3. |
| `twodts-phan-tich.ipynb` | Đánh giá ngoài phân phối trên NF-ToN-IoT-v3, phân tích độ nhạy đặc trưng và giải thích dự đoán. |
| `b-twodts-ablation.ipynb` | So sánh bốn cấu hình cắt bỏ để đo vai trò của từng thành phần kiến trúc. |

Các notebook được giữ nguyên mã và đầu ra của lần chạy đã dùng để tổng hợp báo
cáo. Vì vậy, nên xem chúng như bản ghi thực nghiệm thay vì chỉnh sửa trực tiếp.

Để thực thi đầy đủ, cần có dữ liệu, checkpoint và môi trường thí nghiệm trong
gói Google Drive đi kèm. Các tài sản này không có trong kho GitHub công khai.
