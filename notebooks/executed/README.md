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
| `twodts-phan-tich.ipynb` | Đánh giá ngoài phân phối trên NF-ToN-IoT-v3, hiệu chuẩn bằng nhãn của một nửa tập đích, phân tích độ nhạy đặc trưng và chạy GNNExplainer. |
| `b-twodts-ablation.ipynb` | So sánh bốn cấu hình cắt bỏ trong cùng một phiên chạy trên NF-UNSW-NB15-v3. |

Các notebook được giữ nguyên mã và đầu ra của lần chạy đã dùng để tổng hợp báo
cáo. Hãy xem chúng như bản ghi thực nghiệm. Không chạy lại hoặc chỉnh sửa trực
tiếp vì các ô ghi kết quả có thể ghi đè đầu ra đã dùng trong báo cáo.

Để đối chiếu đầy đủ, cần `data/` và `models/` trong gói Google Drive đi kèm.
Môi trường Python được mô tả bằng `requirements.txt` trong kho GitHub.
