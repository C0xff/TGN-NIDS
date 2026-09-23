# docs — tài liệu kỹ thuật và kết quả đầu ra

Thư mục này tập hợp số liệu đầu ra và các tài liệu kỹ thuật dùng để giải thích,
đối chiếu kết quả của đồ án.

```
docs/
├── SO_LIEU_DAU_RA_CHI_TIET.md   từng con số kèm đủ điều kiện đo
├── KET_QUA_DO_LUONG.md          bảng tổng hợp kết quả và mốc đối sánh
├── THUC_NGHIEM_CAT_BO_KIEN_TRUC.md   bốn cấu hình cắt bỏ
├── KIEM_CHUNG_TAI_LIEU_THAM_KHAO.md  đối chiếu số dẫn lại với bản toàn văn
├── CO_SO_LY_THUYET_VA_DOI_SANH.md    cơ sở lý thuyết và điều kiện đối sánh
├── BO_CONG_CU_GIAI_THICH.md          ba công cụ giải thích
├── QUY_TAC_SINH_TAP_DEMO.md          nguồn và giao thức ba tệp trình diễn
├── TEP_TRONG_SO_CHAY_DUOC_TREN_MAY_KHAC.md  thành phần checkpoint
└── SO_DO_KIEN_TRUC_DU_AN.md          sơ đồ mã, dữ liệu và mô hình
```

## Số liệu đầu ra

| Tệp | Nội dung |
|---|---|
| `SO_LIEU_DAU_RA_CHI_TIET.md` | Từng con số đo được kèm đủ điều kiện đo: bộ dữ liệu, bài toán, tập đo, khoá JSON trong tệp kết quả |
| `KET_QUA_DO_LUONG.md` | Bảng tổng hợp toàn bộ thí nghiệm và các mốc đối sánh |

Hai tệp này **sinh tự động, đừng sửa tay**:

```
python scripts/analysis/sinh_so_lieu_dau_ra.py
python scripts/analysis/tong_hop_ket_qua.py
```

Kiểm tính đúng bằng cách chạy lại rồi so tệp; hai bản phải giống từng byte.

## Tài liệu chuyên đề

| Tệp | Nội dung |
|---|---|
| `THUC_NGHIEM_CAT_BO_KIEN_TRUC.md` | Bốn cấu hình cắt bỏ, cách vô hiệu hoá từng thành phần, giới hạn phát hiện, và thiết kế bắt cặp nhiều hạt giống thuộc Hướng phát triển |
| `KIEM_CHUNG_TAI_LIEU_THAM_KHAO.md` | Kết quả đối chiếu từng số liệu dẫn lại với bản toàn văn của bài gốc |
| `CO_SO_LY_THUYET_VA_DOI_SANH.md` | Cơ sở lý thuyết, điều kiện đối sánh, và ánh xạ đối tượng nghiên cứu của đề cương sang tên lớp của bộ dữ liệu |
| `BO_CONG_CU_GIAI_THICH.md` | Ba công cụ giải thích, câu hỏi mỗi công cụ trả lời, và cách đọc kết quả |
| `QUY_TAC_SINH_TAP_DEMO.md` | Nguồn và giao thức của ba tệp trình diễn trong `data/samples/` |
| `TEP_TRONG_SO_CHAY_DUOC_TREN_MAY_KHAC.md` | Thành phần checkpoint cần có để nạp lại được trên máy khác |
| `SO_DO_KIEN_TRUC_DU_AN.md` | Sơ đồ cấu trúc mã, dữ liệu và mô hình |

## Nguồn sự thật về số liệu

Khi hai nguồn chênh nhau, thứ tự ưu tiên là: notebook đã thực thi trong
`notebooks/executed/`, rồi kết quả và checkpoint trong `models/saved/`, rồi
notebook nguồn, sau cùng mới đến báo cáo và tài liệu diễn giải.

Tài liệu diễn giải không được ghi đè số liệu thực nghiệm.
