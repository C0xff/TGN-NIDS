# SỐ LIỆU ĐẦU RA CỦA MÔ HÌNH: MÔ TẢ ĐẦY ĐỦ TỪNG CHỈ SỐ

**Tệp này sinh tự động. Đừng sửa tay.** Dựng lại bằng:

```
python scripts/analysis/sinh_so_lieu_dau_ra.py
```

Script đọc đúng những tệp sau và không đọc gì khác:

- `models/saved/twoDTS_train_v2/07_giai_doan_mot/models/result_stage1_nf_unsw_nb15_v2.json`
- `models/saved/twoDTS_train_v2/08_giai_doan_hai/models/result_stage2_nf_unsw_nb15_v2.json`
- `models/saved/twoDTS_train_v3/07_giai_doan_mot/models/result_stage1_nf_unsw_nb15_v3.json`
- `models/saved/twoDTS_train_v3/07_giai_doan_mot/models/result_stage1_nf_cse_cic_ids2018_v3.json`
- `models/saved/twoDTS_train_v3/08_giai_doan_hai/models/result_stage2_nf_unsw_nb15_v3.json`
- `models/saved/twoDTS_train_v3/08_giai_doan_hai/models/result_stage2_nf_cse_cic_ids2018_v3.json`
- `models/saved/twoDTS_train_v2/02_ho_so_du_lieu/tables/ho_so_du_lieu.csv`
- `models/saved/twoDTS_train_v3/02_ho_so_du_lieu/tables/ho_so_du_lieu.csv`
- `models/saved/twoDTS_phan_tich/metrics/02_ngoai_phan_phoi/ngoai_phan_phoi.json`
- `models/saved/twoDTS_phan_tich/tables/04_giai_thich/do_quan_trong_dac_trung.csv`
- `models/saved/twoDTS_phan_tich/tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv`
- `models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv`
- `models/saved/twoDTS_demo/metrics/do_tap_demo.json`

Kiểm tính chính xác bằng cách chạy lại script rồi so tệp sinh ra với tệp đang
có; hai bản phải giống từng byte. Mọi con số đều là giá trị đọc thẳng từ tệp
nguồn, hoặc là phép chia và phép trừ trên các giá trị đó; không con số nào gõ tay.

Mục đích: mỗi con số đi kèm **đủ điều kiện đo**, để không ai, người hay công cụ,
trích nó sang một bối cảnh khác.

Một chỉ số chỉ được trích vào báo cáo khi nêu đủ bốn thành phần: **bộ dữ liệu**,
**bài toán**, **tập đo**, và **tên chỉ số đúng như định nghĩa dưới đây**. Thiếu
một trong bốn thì con số đó không được dùng.

## 0. Bảng tra nhanh sáu cấu hình

| Cấu hình | Bộ dữ liệu | Bài toán | Giao thức chia tập | Chỉ số chính | Giá trị |
|---|---|---|---|---|---:|
| `stage1_nf_unsw_nb15_v2` | NF-UNSW-NB15-v2 | hai lớp | `graphids` | F1 lớp tấn công | 67,092% |
| `stage2_nf_unsw_nb15_v2` | NF-UNSW-NB15-v2 | nhiều lớp | `graphids` | Macro F1 | 53,340% |
| `stage1_nf_unsw_nb15_v3` | NF-UNSW-NB15-v3 | hai lớp | `chronological_80_10_10` | F1 lớp tấn công | 97,615% |
| `stage1_nf_cse_cic_ids2018_v3` | NF-CSE-CIC-IDS2018-v3 | hai lớp | `graphids` | F1 lớp tấn công | 23,994% |
| `stage2_nf_unsw_nb15_v3` | NF-UNSW-NB15-v3 | nhiều lớp | `chronological_80_10_10` | Macro F1 | 55,961% |
| `stage2_nf_cse_cic_ids2018_v3` | NF-CSE-CIC-IDS2018-v3 | nhiều lớp | `graphids` | Macro F1 | 56,372% |

Giai đoạn một báo cáo **F1 của lớp tấn công**, giai đoạn hai báo cáo **Macro F1**.
Hai chỉ số này đo hai bài toán khác nhau nên **không đặt cạnh nhau như thể so
sánh được**, và không cộng trừ với nhau.

## `stage1_nf_unsw_nb15_v2`

**Nguồn:** `models/saved/twoDTS_train_v2/07_giai_doan_mot/models/result_stage1_nf_unsw_nb15_v2.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-UNSW-NB15-v2 |
| Giai đoạn | một (self_supervised) |
| Bài toán | hai lớp, tấn công so với lành tính |
| Giao thức chia tập | `graphids` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 2.390.275 luồng |
| Số luồng thực sự đưa vào runner | 2.390.275 |
| Tỷ lệ so với bộ gốc | 100,000% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.836.177 luồng, 0 tấn công (0,000%) |
| Tập kiểm định | 239.027 luồng, 9.505 tấn công (3,977%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 239.031 luồng, 9.508 tấn công (3,978%) |
| Số đỉnh đồ thị | 44 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 41731,30 luồng |
| Số chiều đặc trưng cạnh | 41 |
| Số tham số mô hình | 1.517.929 |
| Tiêu chí dừng sớm | `pr_auc`, patience 25 |
| Vòng tốt nhất | 6 trên ngân sách 150 |
| Ngưỡng quyết định | 0,459939 |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 24,4 phút |
| Độ trễ suy luận | 1,858 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 19:04:56 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 97,537% | tỷ lệ đúng trên cả hai lớp, bị lớp lành tính chi phối nên không dùng để kết luận | `metrics.accuracy` |
| F1 lớp tấn công | 67,092% | chỉ số chính của giai đoạn một, chỉ nói về lớp tấn công | `metrics.f1` |
| Macro F1 | 82,906% | trung bình F1 của hai lớp, không trọng số theo số mẫu | `metrics.f1_macro` |
| Precision lớp tấn công | 71,602% | trong số luồng bị gắn cờ, bao nhiêu phần đúng là tấn công | `metrics.precision` |
| Recall lớp tấn công | 63,115% | trong số tấn công thật, bao nhiêu phần bị bắt; chỉ số này phải luôn đứng cạnh Macro F1 | `metrics.recall` |
| Tỷ lệ báo động giả | 1,037% | luồng lành tính bị gắn cờ chia tổng luồng lành tính, bằng FPR | `metrics.far` |
| PR-AUC | 66,513% | diện tích dưới đường Precision-Recall, không phụ thuộc ngưỡng | `metrics.pr_auc` |
| ROC-AUC | 94,954% | diện tích dưới ROC, lạc quan khi dữ liệu lệch lớp nặng | `metrics.roc_auc` |
| Mốc phân loại tầm thường | 7,651% | F1 mà một bộ gán nhãn tấn công cho mọi luồng sẽ đạt, tính bằng 2r/(1+r) với r là tỷ lệ tấn công của tập kiểm thử | `tính từ split_stats.test` |

Ma trận nhầm lẫn trên tập kiểm thử: TP 6.001, FP 2.380, TN 227.143, FN 3.507.

F1 lớp tấn công 67,092% so với mốc tầm thường 7,651%: **vượt mốc**, chênh 59,440 điểm phần trăm.

## `stage2_nf_unsw_nb15_v2`

**Nguồn:** `models/saved/twoDTS_train_v2/08_giai_doan_hai/models/result_stage2_nf_unsw_nb15_v2.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-UNSW-NB15-v2 |
| Giai đoạn | hai (supervised) |
| Bài toán | nhiều lớp, 10 lớp, lớp lành tính là `Benign` |
| Giao thức chia tập | `graphids` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 2.390.275 luồng |
| Số luồng thực sự đưa vào runner | 2.390.275 |
| Tỷ lệ so với bộ gốc | 100,000% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.912.217 luồng, 76.040 tấn công (3,977%) |
| Tập kiểm định | 239.027 luồng, 9.505 tấn công (3,977%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 239.031 luồng, 9.508 tấn công (3,978%) |
| Số đỉnh đồ thị | 44 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 43459,48 luồng |
| Số chiều đặc trưng cạnh | 41 |
| Số tham số mô hình | 121.162 |
| Tiêu chí dừng sớm | `f1_macro`, patience 15 |
| Vòng tốt nhất | 31 trên ngân sách 100 |
| Ngưỡng quyết định | không dùng, bài toán nhiều lớp chọn lớp có xác suất cao nhất |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 30,4 phút |
| Độ trễ suy luận | 1,206 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 19:36:10 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 98,373% | tỷ lệ đúng trên toàn bộ lớp, bị lớp lành tính chi phối | `metrics.accuracy` |
| Macro F1 | 53,340% | chỉ số chính của giai đoạn hai, trung bình không trọng số trên mọi lớp, nên một lớp hiếm nặng bằng một lớp lớn | `metrics.f1_macro` |
| Macro Precision | 53,409% | trung bình không trọng số precision theo lớp | `metrics.precision_macro` |
| Macro Recall | 59,629% | trung bình không trọng số recall theo lớp, **không phải** recall của lớp tấn công | `metrics.recall_macro` |
| Weighted F1 | 98,498% | có trọng số theo số mẫu nên gần như chỉ phản ánh lớp lành tính | `metrics.f1_weighted` |
| Macro FAR | 0,270% | trung bình tỷ lệ báo động giả theo lớp | `metrics.far_macro` |
| Recall lớp tấn công, quy về hai lớp | 98,896% | gộp mọi lớp tấn công thành một rồi đo recall; đây là con số phải đứng cạnh Macro F1 | `metrics.binary_equivalent.recall` |
| Tỷ lệ báo động giả, quy về hai lớp | 0,465% | đo sau khi quy về hai lớp, không so sánh được với Macro FAR | `metrics.binary_equivalent.far` |

Cấu hình này **không có** `roc_auc` và `pr_auc`, vì hai chỉ số đó chỉ định
nghĩa cho bài toán hai lớp. Thấy chúng đứng cạnh Macro F1 của cấu hình này
thì nghĩa là chúng bị lấy từ bài toán hai lớp quy đổi, và đó là lỗi gắn chỉ
số sai bài toán.

**Chỉ số theo từng lớp, đo trên tập kiểm thử**

| Lớp | Precision | Recall | F1 | FAR | Số mẫu |
|---|---:|---:|---:|---:|---:|
| Analysis | 13,636% | 7,826% | 9,945% | 0,048% | 230 |
| Backdoor | 13,025% | 71,429% | 22,033% | 0,433% | 217 |
| Benign | 99,954% | 99,535% | 99,744% | 1,104% | 229.523 |
| DoS | 16,043% | 10,345% | 12,579% | 0,132% | 580 |
| Exploits | 73,738% | 80,070% | 76,774% | 0,382% | 3.156 |
| Fuzzers | 68,131% | 70,910% | 69,493% | 0,312% | 2.231 |
| Generic | 93,498% | 72,947% | 81,954% | 0,035% | 1.656 |
| Reconnaissance | 68,154% | 78,873% | 73,123% | 0,198% | 1.278 |
| Shellcode | 50,407% | 86,713% | 63,753% | 0,051% | 143 |
| Worms | 37,500% | 17,647% | 24,000% | 0,002% | 17 |

## `stage1_nf_unsw_nb15_v3`

**Nguồn:** `models/saved/twoDTS_train_v3/07_giai_doan_mot/models/result_stage1_nf_unsw_nb15_v3.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-UNSW-NB15-v3 |
| Giai đoạn | một (self_supervised) |
| Bài toán | hai lớp, tấn công so với lành tính |
| Giao thức chia tập | `chronological_80_10_10` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 2.365.424 luồng |
| Số luồng thực sự đưa vào runner | 2.365.424 |
| Tỷ lệ so với bộ gốc | 100,000% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.806.038 luồng, 0 tấn công (0,000%) |
| Tập kiểm định | 236.542 luồng, 21.296 tấn công (9,003%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 236.543 luồng, 20.096 tấn công (8,496%) |
| Số đỉnh đồ thị | 44 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 41046,32 luồng |
| Số chiều đặc trưng cạnh | 49 |
| Số tham số mô hình | 1.539.441 |
| Tiêu chí dừng sớm | `pr_auc`, patience 25 |
| Vòng tốt nhất | 12 trên ngân sách 150 |
| Ngưỡng quyết định | 0,771393 |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 26,7 phút |
| Độ trễ suy luận | 1,78 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 19:07:48 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 99,590% | tỷ lệ đúng trên cả hai lớp, bị lớp lành tính chi phối nên không dùng để kết luận | `metrics.accuracy` |
| F1 lớp tấn công | 97,615% | chỉ số chính của giai đoạn một, chỉ nói về lớp tấn công | `metrics.f1` |
| Macro F1 | 98,695% | trung bình F1 của hai lớp, không trọng số theo số mẫu | `metrics.f1_macro` |
| Precision lớp tấn công | 96,467% | trong số luồng bị gắn cờ, bao nhiêu phần đúng là tấn công | `metrics.precision` |
| Recall lớp tấn công | 98,791% | trong số tấn công thật, bao nhiêu phần bị bắt; chỉ số này phải luôn đứng cạnh Macro F1 | `metrics.recall` |
| Tỷ lệ báo động giả | 0,336% | luồng lành tính bị gắn cờ chia tổng luồng lành tính, bằng FPR | `metrics.far` |
| PR-AUC | 99,224% | diện tích dưới đường Precision-Recall, không phụ thuộc ngưỡng | `metrics.pr_auc` |
| ROC-AUC | 99,954% | diện tích dưới ROC, lạc quan khi dữ liệu lệch lớp nặng | `metrics.roc_auc` |
| Mốc phân loại tầm thường | 15,661% | F1 mà một bộ gán nhãn tấn công cho mọi luồng sẽ đạt, tính bằng 2r/(1+r) với r là tỷ lệ tấn công của tập kiểm thử | `tính từ split_stats.test` |

Ma trận nhầm lẫn trên tập kiểm thử: TP 19.853, FP 727, TN 215.720, FN 243.

F1 lớp tấn công 97,615% so với mốc tầm thường 15,661%: **vượt mốc**, chênh 81,954 điểm phần trăm.

## `stage1_nf_cse_cic_ids2018_v3`

**Nguồn:** `models/saved/twoDTS_train_v3/07_giai_doan_mot/models/result_stage1_nf_cse_cic_ids2018_v3.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-CSE-CIC-IDS2018-v3 |
| Giai đoạn | một (self_supervised) |
| Bài toán | hai lớp, tấn công so với lành tính |
| Giao thức chia tập | `graphids` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 20.115.529 luồng |
| Số luồng thực sự đưa vào runner | 2.365.423 |
| Tỷ lệ so với bộ gốc | 11,759% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.647.663 luồng, 0 tấn công (0,000%) |
| Tập kiểm định | 236.542 luồng, 30.584 tấn công (12,930%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 236.549 luồng, 30.591 tấn công (12,932%) |
| Số đỉnh đồ thị | 76.007 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 21,68 luồng |
| Số chiều đặc trưng cạnh | 49 |
| Số tham số mô hình | 1.539.441 |
| Tiêu chí dừng sớm | `pr_auc`, patience 25 |
| Vòng tốt nhất | 39 trên ngân sách 150 |
| Ngưỡng quyết định | 1,020348 |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 113,3 phút |
| Độ trễ suy luận | 5,138 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 21:03:09 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 77,971% | tỷ lệ đúng trên cả hai lớp, bị lớp lành tính chi phối nên không dùng để kết luận | `metrics.accuracy` |
| F1 lớp tấn công | 23,994% | chỉ số chính của giai đoạn một, chỉ nói về lớp tấn công | `metrics.f1` |
| Macro F1 | 55,556% | trung bình F1 của hai lớp, không trọng số theo số mẫu | `metrics.f1_macro` |
| Precision lớp tấn công | 21,663% | trong số luồng bị gắn cờ, bao nhiêu phần đúng là tấn công | `metrics.precision` |
| Recall lớp tấn công | 26,887% | trong số tấn công thật, bao nhiêu phần bị bắt; chỉ số này phải luôn đứng cạnh Macro F1 | `metrics.recall` |
| Tỷ lệ báo động giả | 14,441% | luồng lành tính bị gắn cờ chia tổng luồng lành tính, bằng FPR | `metrics.far` |
| PR-AUC | 18,147% | diện tích dưới đường Precision-Recall, không phụ thuộc ngưỡng | `metrics.pr_auc` |
| ROC-AUC | 64,701% | diện tích dưới ROC, lạc quan khi dữ liệu lệch lớp nặng | `metrics.roc_auc` |
| Mốc phân loại tầm thường | 22,903% | F1 mà một bộ gán nhãn tấn công cho mọi luồng sẽ đạt, tính bằng 2r/(1+r) với r là tỷ lệ tấn công của tập kiểm thử | `tính từ split_stats.test` |

Ma trận nhầm lẫn trên tập kiểm thử: TP 8.225, FP 29.743, TN 176.215, FN 22.366.

F1 lớp tấn công 23,994% so với mốc tầm thường 22,903%: **vượt mốc**, chênh 1,091 điểm phần trăm.

## `stage2_nf_unsw_nb15_v3`

**Nguồn:** `models/saved/twoDTS_train_v3/08_giai_doan_hai/models/result_stage2_nf_unsw_nb15_v3.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-UNSW-NB15-v3 |
| Giai đoạn | hai (supervised) |
| Bài toán | nhiều lớp, 10 lớp, lớp lành tính là `Benign` |
| Giao thức chia tập | `chronological_80_10_10` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 2.365.424 luồng |
| Số luồng thực sự đưa vào runner | 2.365.424 |
| Tỷ lệ so với bộ gốc | 100,000% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.892.339 luồng, 86.301 tấn công (4,561%) |
| Tập kiểm định | 236.542 luồng, 21.296 tấn công (9,003%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 236.543 luồng, 20.096 tấn công (8,496%) |
| Số đỉnh đồ thị | 44 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 43007,70 luồng |
| Số chiều đặc trưng cạnh | 49 |
| Số tham số mô hình | 122.698 |
| Tiêu chí dừng sớm | `f1_macro`, patience 15 |
| Vòng tốt nhất | 58 trên ngân sách 100 |
| Ngưỡng quyết định | không dùng, bài toán nhiều lớp chọn lớp có xác suất cao nhất |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 43,7 phút |
| Độ trễ suy luận | 1,16 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 21:47:38 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 97,136% | tỷ lệ đúng trên toàn bộ lớp, bị lớp lành tính chi phối | `metrics.accuracy` |
| Macro F1 | 55,961% | chỉ số chính của giai đoạn hai, trung bình không trọng số trên mọi lớp, nên một lớp hiếm nặng bằng một lớp lớn | `metrics.f1_macro` |
| Macro Precision | 60,637% | trung bình không trọng số precision theo lớp | `metrics.precision_macro` |
| Macro Recall | 61,262% | trung bình không trọng số recall theo lớp, **không phải** recall của lớp tấn công | `metrics.recall_macro` |
| Weighted F1 | 96,979% | có trọng số theo số mẫu nên gần như chỉ phản ánh lớp lành tính | `metrics.f1_weighted` |
| Macro FAR | 0,296% | trung bình tỷ lệ báo động giả theo lớp | `metrics.far_macro` |
| Recall lớp tấn công, quy về hai lớp | 99,955% | gộp mọi lớp tấn công thành một rồi đo recall; đây là con số phải đứng cạnh Macro F1 | `metrics.binary_equivalent.recall` |
| Tỷ lệ báo động giả, quy về hai lớp | 0,052% | đo sau khi quy về hai lớp, không so sánh được với Macro FAR | `metrics.binary_equivalent.far` |

Cấu hình này **không có** `roc_auc` và `pr_auc`, vì hai chỉ số đó chỉ định
nghĩa cho bài toán hai lớp. Thấy chúng đứng cạnh Macro F1 của cấu hình này
thì nghĩa là chúng bị lấy từ bài toán hai lớp quy đổi, và đó là lỗi gắn chỉ
số sai bài toán.

**Chỉ số theo từng lớp, đo trên tập kiểm thử**

| Lớp | Precision | Recall | F1 | FAR | Số mẫu |
|---|---:|---:|---:|---:|---:|
| Analysis | 33,438% | 75,352% | 46,320% | 0,090% | 142 |
| Backdoor | 71,429% | 0,406% | 0,808% | 0,001% | 1.231 |
| Benign | 99,996% | 99,948% | 99,972% | 0,045% | 216.447 |
| DoS | 20,813% | 22,563% | 21,653% | 0,273% | 749 |
| Exploits | 65,714% | 68,104% | 66,888% | 1,006% | 6.515 |
| Fuzzers | 63,651% | 83,461% | 72,222% | 1,107% | 5.369 |
| Generic | 88,013% | 78,257% | 82,849% | 0,139% | 3.040 |
| Reconnaissance | 74,864% | 59,761% | 66,465% | 0,237% | 2.761 |
| Shellcode | 60,248% | 72,388% | 65,763% | 0,054% | 268 |
| Worms | 28,205% | 52,381% | 36,667% | 0,012% | 21 |

## `stage2_nf_cse_cic_ids2018_v3`

**Nguồn:** `models/saved/twoDTS_train_v3/08_giai_doan_hai/models/result_stage2_nf_cse_cic_ids2018_v3.json`

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu | NF-CSE-CIC-IDS2018-v3 |
| Giai đoạn | hai (supervised) |
| Bài toán | nhiều lớp, 15 lớp, lớp lành tính là `Benign` |
| Giao thức chia tập | `graphids` |
| Quy mô bộ gốc, trước mọi phép rút mẫu | 20.115.529 luồng |
| Số luồng thực sự đưa vào runner | 2.365.423 |
| Tỷ lệ so với bộ gốc | 11,759% |
| Trường `sample_ratio` trong tệp kết quả | 100%, đây là phép rút mẫu do runner thực hiện, không tính phép rút mẫu làm trước đó |
| Tập huấn luyện | 1.892.332 luồng, 244.669 tấn công (12,929%) |
| Tập kiểm định | 236.542 luồng, 30.584 tấn công (12,930%) |
| **Tập kiểm thử, nơi mọi chỉ số dưới đây được đo** | 236.549 luồng, 30.591 tấn công (12,932%) |
| Số đỉnh đồ thị | 76.060 |
| Mật độ luồng huấn luyện trên mỗi đỉnh | 24,88 luồng |
| Số chiều đặc trưng cạnh | 49 |
| Số tham số mô hình | 123.663 |
| Tiêu chí dừng sớm | `f1_macro`, patience 15 |
| Vòng tốt nhất | 24 trên ngân sách 100 |
| Ngưỡng quyết định | không dùng, bài toán nhiều lớp chọn lớp có xác suất cao nhất |
| Thiết bị | Tesla T4, torch 2.10.0+cu128 |
| Thời gian huấn luyện | 65,9 phút |
| Độ trễ suy luận | 4,574 ms trên 100 luồng |
| Ablation | node_identity `False`, features `rỗng` |
| Phiên bản mã nguồn | 1.1.0 |
| Thời điểm chạy xong | 2026-09-04 22:55:33 |

**Chỉ số trên tập kiểm thử**

| Chỉ số | Giá trị | Ý nghĩa và giới hạn diễn giải | Khoá JSON |
|---|---:|---|---|
| Accuracy | 95,161% | tỷ lệ đúng trên toàn bộ lớp, bị lớp lành tính chi phối | `metrics.accuracy` |
| Macro F1 | 56,372% | chỉ số chính của giai đoạn hai, trung bình không trọng số trên mọi lớp, nên một lớp hiếm nặng bằng một lớp lớn | `metrics.f1_macro` |
| Macro Precision | 63,956% | trung bình không trọng số precision theo lớp | `metrics.precision_macro` |
| Macro Recall | 53,867% | trung bình không trọng số recall theo lớp, **không phải** recall của lớp tấn công | `metrics.recall_macro` |
| Weighted F1 | 94,706% | có trọng số theo số mẫu nên gần như chỉ phản ánh lớp lành tính | `metrics.f1_weighted` |
| Macro FAR | 1,404% | trung bình tỷ lệ báo động giả theo lớp | `metrics.far_macro` |
| Recall lớp tấn công, quy về hai lớp | 81,374% | gộp mọi lớp tấn công thành một rồi đo recall; đây là con số phải đứng cạnh Macro F1 | `metrics.binary_equivalent.recall` |
| Tỷ lệ báo động giả, quy về hai lớp | 0,034% | đo sau khi quy về hai lớp, không so sánh được với Macro FAR | `metrics.binary_equivalent.far` |

Cấu hình này **không có** `roc_auc` và `pr_auc`, vì hai chỉ số đó chỉ định
nghĩa cho bài toán hai lớp. Thấy chúng đứng cạnh Macro F1 của cấu hình này
thì nghĩa là chúng bị lấy từ bài toán hai lớp quy đổi, và đó là lỗi gắn chỉ
số sai bài toán.

**Chỉ số theo từng lớp, đo trên tập kiểm thử**

| Lớp | Precision | Recall | F1 | FAR | Số mẫu |
|---|---:|---:|---:|---:|---:|
| Benign | 97,307% | 99,966% | 98,618% | 18,626% | 205.958 |
| Bot | 99,349% | 100,000% | 99,674% | 0,007% | 2.443 |
| Brute_Force_-Web | 0,000% | 0,000% | 0,000% | 2,392% | 19 |
| Brute_Force_-XSS | 0,000% | 0,000% | 0,000% | 0,000% | 6 |
| DDOS_attack-HOIC | 99,989% | 74,390% | 85,311% | 0,000% | 12.140 |
| DDOS_attack-LOIC-UDP | 85,417% | 100,000% | 92,135% | 0,003% | 41 |
| DDoS_attacks-LOIC-HTTP | 100,000% | 96,906% | 98,429% | 0,000% | 3.394 |
| DoS_attacks-GoldenEye | 100,000% | 77,254% | 87,167% | 0,000% | 721 |
| DoS_attacks-Hulk | 100,000% | 100,000% | 100,000% | 0,000% | 1.177 |
| DoS_attacks-SlowHTTPTest | 0,000% | 0,000% | 0,000% | 0,000% | 1.242 |
| DoS_attacks-Slowloris | 96,209% | 47,877% | 63,937% | 0,003% | 424 |
| FTP-BruteForce | 0,000% | 0,000% | 0,000% | 0,000% | 4.548 |
| Infilteration | 81,073% | 11,613% | 20,316% | 0,026% | 2.213 |
| SQL_Injection | 0,000% | 0,000% | 0,000% | 0,000% | 6 |
| SSH-Bruteforce | 100,000% | 100,000% | 100,000% | 0,000% | 2.217 |

## Phân tích ngoài phân phối trên NF-ToN-IoT-v3

**Nguồn:** `models/saved/twoDTS_phan_tich/metrics/02_ngoai_phan_phoi/ngoai_phan_phoi.json`

Đây là phép đo **khái quát sang bộ dữ liệu chưa từng huấn luyện**. Mô hình giữ
nguyên trọng số, chỉ đổi dữ liệu đầu vào, nên mọi con số dưới đây nói về khả năng
chuyển miền, **không** so sánh được với chỉ số trên tập kiểm thử cùng phân phối.

| Điều kiện đo | Giá trị |
|---|---|
| Bộ dữ liệu đích | NF-ToN-IoT-v3 |
| Số luồng lấy mẫu | 250.000 |
| Hạt giống lấy mẫu / chia | 42 / 0 |
| Giao thức | chọn ngưỡng trên nửa A, chấm điểm trên nửa B |

| Mô hình gốc | Tỷ lệ dương | Mốc tầm thường | F1 ngưỡng mặc định | F1 sau hiệu chuẩn | Recall | ROC-AUC | PR-AUC | Vượt mốc |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `stage2_nf_cse_cic_ids2018_v3` | 38,982% | 56,097% | 0,883% | 69,463% | 82,275% | 77,206% | 59,076% | có |
| `stage2_nf_unsw_nb15_v2` | 38,982% | 56,097% | 0,880% | 56,097% | 100,000% | 62,587% | 47,940% | **không** |
| `stage2_nf_unsw_nb15_v3` | 38,982% | 56,097% | 41,791% | 74,073% | 82,390% | 82,583% | 67,886% | có |

Cột **F1 ngưỡng mặc định** là F1 khi áp thẳng ngưỡng đã hiệu chuẩn trên bộ gốc;
cột **F1 sau hiệu chuẩn** là F1 khi chọn lại ngưỡng trên nửa A của bộ đích rồi
chấm trên nửa B. Hai cột này **không được trộn**: cột thứ hai cần nhãn của bộ
đích, tức một điều kiện mà tình huống triển khai thật không có.

## Giải thích mô hình: độ quan trọng đặc trưng

**Nguồn:** `models/saved/twoDTS_phan_tich/tables/04_giai_thich/do_quan_trong_dac_trung.csv`

Phép triệt tiêu từng đặc trưng: lần lượt đặt một cột về giá trị không rồi đo độ
lệch của xác suất tấn công. Đây là **công cụ giải thích**, không phải ablation
kiến trúc, và không nói gì về đóng góp của thành phần kiến trúc nào.

**`stage2_nf_cse_cic_ids2018_v3`**, năm đặc trưng dẫn đầu trong 49 đặc trưng đã đo:

| Hạng | Đặc trưng | Độ quan trọng tương đối | Độ lệch xác suất tuyệt đối trung bình |
|---:|---|---:|---:|
| 1 | `PROTOCOL` | 12,404% | 3,086% |
| 2 | `L4_DST_PORT` | 8,209% | 2,042% |
| 3 | `SHORTEST_FLOW_PKT` | 8,092% | 2,013% |
| 4 | `MIN_IP_PKT_LEN` | 8,028% | 1,997% |
| 5 | `L4_SRC_PORT` | 7,189% | 1,789% |

**`stage2_nf_unsw_nb15_v2`**, năm đặc trưng dẫn đầu trong 41 đặc trưng đã đo:

| Hạng | Đặc trưng | Độ quan trọng tương đối | Độ lệch xác suất tuyệt đối trung bình |
|---:|---|---:|---:|
| 1 | `SHORTEST_FLOW_PKT` | 15,591% | 7,363% |
| 2 | `TCP_WIN_MAX_IN` | 10,826% | 5,113% |
| 3 | `TCP_WIN_MAX_OUT` | 9,924% | 4,687% |
| 4 | `MIN_IP_PKT_LEN` | 9,613% | 4,540% |
| 5 | `PROTOCOL` | 9,208% | 4,349% |

**`stage2_nf_unsw_nb15_v3`**, năm đặc trưng dẫn đầu trong 49 đặc trưng đã đo:

| Hạng | Đặc trưng | Độ quan trọng tương đối | Độ lệch xác suất tuyệt đối trung bình |
|---:|---|---:|---:|
| 1 | `SHORTEST_FLOW_PKT` | 37,125% | 59,501% |
| 2 | `DST_TO_SRC_IAT_MIN` | 13,127% | 21,038% |
| 3 | `SERVER_TCP_FLAGS` | 5,066% | 8,119% |
| 4 | `MIN_IP_PKT_LEN` | 3,850% | 6,171% |
| 5 | `MIN_TTL` | 3,715% | 5,954% |

## Giải thích mô hình: trọng số cạnh học bằng GNNExplainer

**Nguồn:** `models/saved/twoDTS_phan_tich/tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv`

| Đại lượng | Giá trị | Ý nghĩa | Cột CSV |
|---|---:|---|---|
| Mô hình | `stage2_nf_unsw_nb15_v3` | lần chạy được giải thích | `run_name` |
| Số luồng chấm điểm | 600 | quy mô đồ thị con đưa vào | `flows_scored` |
| Xác suất tấn công trước khi che | 99,989% | dự đoán gốc của luồng mục tiêu | `attack_probability_before` |
| Xác suất tấn công sau khi che | 98,647% | sau khi áp mặt nạ cạnh đã học | `attack_probability_after` |
| Trọng số cạnh nhỏ nhất | 10,817% | | `edge_mask_min` |
| Trọng số cạnh trung vị | 12,249% | trung vị thấp nghĩa là phần lớn cạnh bị hạ | `edge_mask_median` |
| Trọng số cạnh lớn nhất | 89,778% | | `edge_mask_max` |
| Độ lệch chuẩn trọng số | 35,360% | càng lớn càng phân biệt được cạnh | `edge_mask_std` |
| Số cạnh giữ trên ngưỡng 0,5 | 189 trên 600 | | `edges_kept_above_half`, `edges_total` |
| Số tham số mô hình bị đổi | 0 | phải bằng 0, phép giải thích không được sửa mô hình | `model_parameters_changed` |
| Số vòng tối ưu mặt nạ | 80 | | `epochs` |

## Thực nghiệm cắt bỏ kiến trúc

**Nguồn:** `models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv`

Mọi giá trị dưới đây đo trên **một hạt giống duy nhất**, nên các chênh lệch
nhỏ chưa tách khỏi dao động giữa các lần chạy và phải trích kèm giới hạn
phát hiện. Mở rộng sang nhiều hạt giống thuộc Hướng phát triển, ghi ở mục 6
`docs/THUC_NGHIEM_CAT_BO_KIEN_TRUC.md`.

Mỗi cấu hình được **huấn luyện lại từ đầu** trên toàn bộ NF-UNSW-NB15-v3, cùng
hạt giống và trong cùng một phiên, nên mốc đối chiếu không lẫn sai khác môi
trường. Trường `question` trong tệp nguồn ghi `retrain`: con số dưới đây trả lời
câu hỏi *kiến trúc thiếu thành phần đó thì học được tới đâu*, **không** trả lời
câu hỏi mô hình đã huấn luyện dựa vào thành phần đó bao nhiêu lúc suy luận.

| Cấu hình | Thành phần bị vô hiệu hoá | Macro F1 | Chênh so với mốc | Recall lớp tấn công | F1 lớp tấn công | Tỷ lệ báo động giả |
|---|---|---:|---:|---:|---:|---:|
| `baseline` | không, đây là mốc đối chiếu | 54,936% | — | 99,980% | 99,923% | 0,012% |
| `random_nodes` | danh tính đỉnh, gán lại ngẫu nhiên đỉnh nguồn và đích | 52,109% | -2,827 điểm | 99,945% | 99,903% | 0,013% |
| `no_iat` | tám cột đặc trưng thời gian IAT | 54,616% | -0,320 điểm | 99,975% | 99,705% | 0,053% |
| `no_propagation` | phép tổng hợp từ đỉnh lân cận, giữ nguyên vector nhớ | 54,444% | -0,492 điểm | 99,980% | 99,873% | 0,022% |

Mốc phân loại tầm thường của tập kiểm thử là 15,661%. Cả bốn cấu hình đều vượt xa mốc
này, nên không cấu hình nào rơi xuống mức đoán bừa.

| Cấu hình | Vòng tốt nhất giai đoạn một | Vòng tốt nhất giai đoạn hai | Thời gian huấn luyện | Bị cắt khi còn đang tiến bộ |
|---|---:|---:|---:|:---:|
| `baseline` | 12 trên 150 | 37 trên 100 | 59,1 phút | không |
| `random_nodes` | 45 trên 150 | 46 trên 100 | 114,0 phút | không |
| `no_iat` | 8 trên 150 | 36 trên 100 | 54,0 phút | không |
| `no_propagation` | 39 trên 150 | 78 trên 100 | 86,7 phút | không |

Cột cuối là phép tự kiểm: vòng tốt nhất nằm sát cuối ngân sách nghĩa là mô hình
vẫn đang tiến bộ khi bị cắt, và con số của nó không đọc được như con số của một
mô hình đã hội tụ. Cả bốn cấu hình đều không rơi vào trường hợp đó.

## Phép đo chức năng trên ba tệp demo

**Nguồn:** `models/saved/twoDTS_demo/metrics/do_tap_demo.json`, dựng lại bằng
`python scripts/analysis/do_tap_demo.py`.

Đây là **phép thử chức năng của giao diện**, không phải kết quả đánh giá mô
hình, và không được dùng để kết luận mô hình tốt hay kém.

Hai trạng thái bộ nhớ khác nhau, và tên hàm dễ gây nhầm: `reset_memory` của
động cơ suy luận **khôi phục** bảng nhớ từ checkpoint chứ không xoá; muốn xoá
trắng phải gọi `memory_module.reset_state()`.

| Tệp | Kịch bản | Trạng thái bộ nhớ | Độ chính xác | Recall lớp tấn công | Tỷ lệ báo động giả | Mốc tầm thường |
|---|---|---|---:|---:|---:|---:|
| `demo_1_cung_phan_phoi.csv` | cùng phân phối với dữ liệu mô hình đã học | checkpoint | 99,940% | 100,000% | 0,066% | 15,668% |
| `demo_1_cung_phan_phoi.csv` | cùng phân phối với dữ liệu mô hình đã học | xoa_trang | 96,870% | 100,000% | 3,421% | 15,668% |
| `demo_2_bo_du_lieu_la.csv` | bộ dữ liệu lạ, mô hình chưa từng huấn luyện trên đó | checkpoint | 74,560% | 5,800% | 15,229% | 22,899% |
| `demo_2_bo_du_lieu_la.csv` | bộ dữ liệu lạ, mô hình chưa từng huấn luyện trên đó | xoa_trang | 76,970% | 21,964% | 14,862% | 22,899% |
| `demo_3_thieu_cot_dia_chi.csv` | thiếu cột địa chỉ, không dựng được đồ thị máy chủ | checkpoint | 59,970% | 4,814% | 32,623% | 21,173% |
| `demo_3_thieu_cot_dia_chi.csv` | thiếu cột địa chỉ, không dựng được đồ thị máy chủ | xoa_trang | 72,190% | 4,392% | 18,705% | 21,173% |

**Hạn chế phải nêu kèm:** Checkpoint lưu bảng nhớ sau khi đã chạy qua tập kiểm thử, nên chỉ số ở trạng thái checkpoint lạc quan hơn tình huống triển khai thật.

