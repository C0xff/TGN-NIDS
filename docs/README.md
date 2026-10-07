# Tài liệu kỹ thuật và kết quả đầu ra (`docs/`)

Thư mục này tập hợp số liệu đầu ra và tài liệu kỹ thuật công khai dùng để giải
thích, đối chiếu kết quả của đồ án. Các tệp ở đây không thay thế dữ liệu, trọng
số hoặc đầu ra gốc; những tài sản đó nằm trong gói Google Drive đi kèm.

## Cấu trúc thư mục

```text
docs/
├── BO_CONG_CU_GIAI_THICH.md
├── CO_SO_LY_THUYET_VA_DOI_SANH.md
├── KET_QUA_DO_LUONG.md
├── KIEM_CHUNG_TAI_LIEU_THAM_KHAO.md
├── QUY_TAC_SINH_TAP_DEMO.md
├── SO_DO_KIEN_TRUC_DU_AN.md
├── SO_LIEU_DAU_RA_CHI_TIET.md
├── TEP_TRONG_SO_CHAY_DUOC_TREN_MAY_KHAC.md
└── THUC_NGHIEM_CAT_BO_KIEN_TRUC.md
```

## Nội dung chính

| Tệp | Nội dung |
| --- | --- |
| `SO_LIEU_DAU_RA_CHI_TIET.md` | Từng số liệu kèm điều kiện đo và nguồn đối chiếu. |
| `KET_QUA_DO_LUONG.md` | Bảng kết quả tổng hợp, thí nghiệm cắt bỏ, các mốc công bố và phép đo bổ trợ của đồ án. |
| `THUC_NGHIEM_CAT_BO_KIEN_TRUC.md` | Thiết kế và kết quả của các cấu hình cắt bỏ. |
| `KIEM_CHUNG_TAI_LIEU_THAM_KHAO.md` | Cách đối chiếu số liệu dẫn lại với nguồn gốc. |
| `CO_SO_LY_THUYET_VA_DOI_SANH.md` | Cơ sở lý thuyết và điều kiện so sánh. |
| `BO_CONG_CU_GIAI_THICH.md` | Các công cụ giải thích và cách đọc kết quả. |
| `QUY_TAC_SINH_TAP_DEMO.md` | Nguồn và giao thức của các tệp trình diễn trong gói dữ liệu nội bộ. |
| `TEP_TRONG_SO_CHAY_DUOC_TREN_MAY_KHAC.md` | Thành phần checkpoint cần có để suy luận trên máy khác. |
| `SO_DO_KIEN_TRUC_DU_AN.md` | Kiến trúc dự án và cách phân tách GitHub với Google Drive. |

## Phạm vi công khai

Các tài liệu này diễn giải notebook đã thực thi và đầu ra thực nghiệm đã chốt.
`KET_QUA_DO_LUONG.md` được dựng từ kết quả JSON, bảng cắt bỏ và phép đo bổ trợ
đã lưu. Số liệu của công trình khác được chép theo tài liệu tham khảo đã kiểm
chứng và được ghi nguồn riêng.
Dữ liệu, checkpoint, kết quả gốc, tài liệu tham khảo và gói Kaggle nằm trong bốn thư
mục `data/`, `models/`, `references/`, `dist/` trên Google Drive. Báo cáo Word
và hồ sơ của trường không thuộc gói được chia sẻ.

Khi cần đối chiếu một giá trị, ưu tiên notebook đã thực thi và đầu ra gốc trong
gói Drive. Không sửa tay số liệu đã công bố chỉ để làm chúng khớp với tài liệu.
