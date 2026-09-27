# Kết quả đo lường

**Tài liệu này được sinh tự động** bởi `scripts/analysis/tong_hop_ket_qua.py`,
đọc thẳng từ các tệp `result_*.json` của lần chạy thật. Không chép tay con
số nào, nên tài liệu luôn khớp dữ liệu gốc. Chạy lại script sau mỗi
notebook để cập nhật.

Các phép đo không sinh ra từ notebook nằm ở cuối, trong mục *Các phép đo
bổ trợ*. Mục đó cũng sinh tự động, đọc từ `models/supplementary/supplementary_measurements.json`
do `scripts/analysis/supplementary_measurements.py` tạo ra và từ bảng tổng hợp thí nghiệm
cắt bỏ. Riêng số liệu của các nghiên cứu đối sánh là trích dẫn, ghi kèm bảng
và điều kiện đo của bài báo gốc. Quy tắc chọn ngưỡng và bảng tham số đã chốt
xem `docs/SO_LIEU_DAU_RA_CHI_TIET.md`.

Số thí nghiệm đã có kết quả: **14**.

## Thư mục `saved`

### Điều kiện chạy

Mọi lần chạy dùng hạt giống 42, GPU Tesla T4 trên Kaggle. Giao thức `chronological_80_10_10` sắp luồng theo `FLOW_START_MILLISECONDS` rồi cắt liên tục 80/10/10. Giao thức `graphids` chia phân tầng theo cột `Attack` với hạt giống 42, cũng theo tỷ lệ 80/10/10. Ở giai đoạn một (`self_supervised`), luồng tấn công bị loại khỏi tập train nên cột Train chỉ đếm luồng Benign. Mọi chỉ số bên dưới đo trên tập test ghi ở cột Test, bộ tiền xử lý khớp trên tập train của chính lần chạy đó.

| Cấu hình | Mã nguồn | Protocol | Task | Mode | Lấy mẫu | Train | Test | Edge features |
|---|---|---|---|---|---:|---:|---:|---:|
| `stage1_nf_cse_cic_ids2018_v3` | 1.1.0 | graphids | binary | self_supervised | 100% | 1.647.663 | 236.549 | 49 |
| `stage1_nf_unsw_nb15_v2` | 1.1.0 | graphids | binary | self_supervised | 100% | 1.836.177 | 239.031 | 41 |
| `stage1_nf_unsw_nb15_v3` | 1.1.0 | chronological_80_10_10 | binary | self_supervised | 100% | 1.806.038 | 236.543 | 49 |
| `stage1_unsw_v3_baseline` | 1.1.0 | chronological_80_10_10 | binary | self_supervised | 100% | 1.806.038 | 236.543 | 49 |
| `stage1_unsw_v3_no_iat` | 1.1.0 | chronological_80_10_10 | binary | self_supervised | 100% | 1.806.038 | 236.543 | 41 |
| `stage1_unsw_v3_no_propagation` | 1.1.0 | chronological_80_10_10 | binary | self_supervised | 100% | 1.806.038 | 236.543 | 49 |
| `stage1_unsw_v3_random_nodes` | 1.1.0 | chronological_80_10_10 | binary | self_supervised | 100% | 1.806.038 | 236.543 | 49 |
| `stage2_nf_cse_cic_ids2018_v3` | 1.1.0 | graphids | multiclass | supervised | 100% | 1.892.332 | 236.549 | 49 |
| `stage2_nf_unsw_nb15_v2` | 1.1.0 | graphids | multiclass | supervised | 100% | 1.912.217 | 239.031 | 41 |
| `stage2_nf_unsw_nb15_v3` | 1.1.0 | chronological_80_10_10 | multiclass | supervised | 100% | 1.892.339 | 236.543 | 49 |
| `stage2_unsw_v3_baseline` | 1.1.0 | chronological_80_10_10 | multiclass | supervised | 100% | 1.892.339 | 236.543 | 49 |
| `stage2_unsw_v3_no_iat` | 1.1.0 | chronological_80_10_10 | multiclass | supervised | 100% | 1.892.339 | 236.543 | 41 |
| `stage2_unsw_v3_no_propagation` | 1.1.0 | chronological_80_10_10 | multiclass | supervised | 100% | 1.892.339 | 236.543 | 49 |
| `stage2_unsw_v3_random_nodes` | 1.1.0 | chronological_80_10_10 | multiclass | supervised | 100% | 1.892.339 | 236.543 | 49 |

### Huấn luyện

| Cấu hình | Vòng đã chạy | Ngân sách | Vòng tốt nhất | Val loss | Trạng thái | Thời gian |
|---|---:|---:|---:|---:|---|---:|
| `stage1_nf_cse_cic_ids2018_v3` | 64 | 150 | 39 | 14.63232 | đã hội tụ | 6795s |
| `stage1_nf_unsw_nb15_v2` | 31 | 150 | 6 | 0.47241 | đã hội tụ | 1464s |
| `stage1_nf_unsw_nb15_v3` | 37 | 150 | 12 | 2.69708 | đã hội tụ | 1602s |
| `stage1_unsw_v3_baseline` | 37 | 150 | 12 | 2.49066 | đã hội tụ | 1686s |
| `stage1_unsw_v3_no_iat` | 33 | 150 | 8 | 3.28597 | đã hội tụ | 1415s |
| `stage1_unsw_v3_no_propagation` | 64 | 150 | 39 | 1.80769 | đã hội tụ | 2176s |
| `stage1_unsw_v3_random_nodes` | 70 | 150 | 45 | 1.61756 | đã hội tụ | 3920s |
| `stage2_nf_cse_cic_ids2018_v3` | 39 | 100 | 24 | 0.75951 | đã hội tụ | 3956s |
| `stage2_nf_unsw_nb15_v2` | 46 | 100 | 31 | 0.23182 | đã hội tụ | 1826s |
| `stage2_nf_unsw_nb15_v3` | 73 | 100 | 58 | 0.52962 | đã hội tụ | 2621s |
| `stage2_unsw_v3_baseline` | 52 | 100 | 37 | 0.52491 | đã hội tụ | 1860s |
| `stage2_unsw_v3_no_iat` | 51 | 100 | 36 | 0.54589 | đã hội tụ | 1822s |
| `stage2_unsw_v3_no_propagation` | 93 | 100 | 78 | 0.56201 | đã hội tụ | 3026s |
| `stage2_unsw_v3_random_nodes` | 61 | 100 | 46 | 0.56835 | đã hội tụ | 2920s |

### Chỉ số, bài toán binary

Cột **mốc tầm thường** là điểm F1 mà một bộ phân loại gán nhãn tấn công cho mọi mẫu sẽ đạt được, tính bằng `2·rate/(1+rate)` với `rate` là tỷ lệ mẫu tấn công của tập kiểm thử. Mục 2.5 bộ quy tắc dự án buộc mọi chỉ số F1 nhị phân phải đứng cạnh mốc này: thiếu nó, một giá trị F1 cao vẫn có thể nằm dưới mức đoán bừa không cần mô hình.

Điều kiện đo: điểm bất thường là sai số tái thiết của giai đoạn một. Ngưỡng chọn trên tập validation sao cho Macro F1 hai lớp lớn nhất, rồi áp nguyên lên tập test. PR-AUC là average precision, ROC-AUC tính trên điểm bất thường, hai chỉ số này không phụ thuộc ngưỡng. FAR là tỷ lệ luồng Benign bị gán tấn công.

| Cấu hình | f1 | f1_macro | precision | recall | far | pr_auc | roc_auc | accuracy | balanced_accuracy | mốc tầm thường | f1 trừ mốc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_nf_cse_cic_ids2018_v3` | 23,994% | 55,556% | 21,663% | 26,887% | 14,441% | 18,147% | 64,701% | 77,971% | 56,223% | 22,903% | +1,091% |
| `stage1_nf_unsw_nb15_v2` | 67,092% | 82,906% | 71,602% | 63,115% | 1,037% | 66,513% | 94,954% | 97,537% | 81,039% | 7,651% | +59,44% |
| `stage1_nf_unsw_nb15_v3` | 97,615% | 98,695% | 96,467% | 98,791% | 0,336% | 99,224% | 99,954% | 99,59% | 99,227% | 15,661% | +81,954% |
| `stage1_unsw_v3_baseline` | 97,484% | 98,624% | 96,273% | 98,726% | 0,355% | 99,092% | 99,944% | 99,567% | 99,186% | 15,661% | +81,823% |
| `stage1_unsw_v3_no_iat` | 97,364% | 98,558% | 96,409% | 98,338% | 0,34% | 99,042% | 99,932% | 99,548% | 98,999% | 15,661% | +81,703% |
| `stage1_unsw_v3_no_propagation` | 96,52% | 98,094% | 94,324% | 98,821% | 0,552% | 98,406% | 99,904% | 99,395% | 99,134% | 15,661% | +80,859% |
| `stage1_unsw_v3_random_nodes` | 97,069% | 98,397% | 96,022% | 98,139% | 0,377% | 99,124% | 99,946% | 99,496% | 98,881% | 15,661% | +81,408% |

### Chỉ số, bài toán multiclass

Giá trị trung bình macro trên toàn bộ các lớp của từng bộ dữ liệu. Số lớp không giống nhau giữa các cấu hình trong bảng, hiện là 10 hoặc 15, nên hai giá trị macro của hai bộ khác nhau không phải hai phép đo cùng thang.

Cột cuối là **recall của lớp tấn công**, đo trên bài toán nhị phân quy đổi: trong toàn bộ luồng tấn công thật, mô hình gọi đúng tên một lớp tấn công nào đó cho bao nhiêu phần. Con số này bắt buộc đi kèm mọi chỗ hiển thị Macro F1, vì Macro F1 có thể cao nhờ riêng lớp lành tính trong khi bộ phát hiện đang bỏ sót phần lớn số cuộc tấn công.

Điều kiện đo: nhãn dự đoán là lớp có xác suất lớn nhất (argmax), không dùng ngưỡng. Encoder của giai đoạn một được đóng băng, chỉ classification head được huấn luyện trên tập train có nhãn.

| Cấu hình | f1_macro | precision_macro | recall_macro | far_macro | accuracy | recall lớp tấn công (nhị phân quy đổi) |
|---|---|---|---|---|---|---|
| `stage2_nf_cse_cic_ids2018_v3` | 56,372% | 63,956% | 53,867% | 1,404% | 95,161% | 81,374% |
| `stage2_nf_unsw_nb15_v2` | 53,34% | 53,409% | 59,629% | 0,27% | 98,373% | 98,896% |
| `stage2_nf_unsw_nb15_v3` | 55,961% | 60,637% | 61,262% | 0,296% | 97,136% | 99,955% |
| `stage2_unsw_v3_baseline` | 54,936% | 56,392% | 58,505% | 0,297% | 97,109% | 99,98% |
| `stage2_unsw_v3_no_iat` | 54,616% | 58,92% | 56,346% | 0,304% | 97,045% | 99,975% |
| `stage2_unsw_v3_no_propagation` | 54,444% | 56,748% | 57,52% | 0,309% | 96,99% | 99,98% |
| `stage2_unsw_v3_random_nodes` | 52,109% | 55,951% | 56,576% | 0,312% | 96,993% | 99,945% |

**Chi tiết từng lớp, `stage2_nf_cse_cic_ids2018_v3`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Benign | 97,307% | 99,966% | 98,618% | 18,626% | 205.958 |
| Bot | 99,349% | 100% | 99,674% | 0,007% | 2.443 |
| Brute_Force_-Web | 0% | 0% | 0% | 2,392% | 19 |
| Brute_Force_-XSS | 0% | 0% | 0% | 0% | 6 |
| DDOS_attack-HOIC | 99,989% | 74,39% | 85,311% | 0,0004% | 12.140 |
| DDOS_attack-LOIC-UDP | 85,417% | 100% | 92,135% | 0,003% | 41 |
| DDoS_attacks-LOIC-HTTP | 100% | 96,906% | 98,429% | 0% | 3.394 |
| DoS_attacks-GoldenEye | 100% | 77,254% | 87,167% | 0% | 721 |
| DoS_attacks-Hulk | 100% | 100% | 100% | 0% | 1.177 |
| DoS_attacks-SlowHTTPTest | 0% | 0% | 0% | 0% | 1.242 |
| DoS_attacks-Slowloris | 96,209% | 47,877% | 63,937% | 0,003% | 424 |
| FTP-BruteForce | 0% | 0% | 0% | 0% | 4.548 |
| Infilteration | 81,073% | 11,613% | 20,316% | 0,026% | 2.213 |
| SQL_Injection | 0% | 0% | 0% | 0% | 6 |
| SSH-Bruteforce | 100% | 100% | 100% | 0% | 2.217 |

**Chi tiết từng lớp, `stage2_nf_unsw_nb15_v2`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 13,636% | 7,826% | 9,945% | 0,048% | 230 |
| Backdoor | 13,025% | 71,429% | 22,033% | 0,433% | 217 |
| Benign | 99,954% | 99,535% | 99,744% | 1,104% | 229.523 |
| DoS | 16,043% | 10,345% | 12,579% | 0,132% | 580 |
| Exploits | 73,738% | 80,07% | 76,774% | 0,382% | 3.156 |
| Fuzzers | 68,131% | 70,91% | 69,493% | 0,313% | 2.231 |
| Generic | 93,498% | 72,947% | 81,954% | 0,035% | 1.656 |
| Reconnaissance | 68,154% | 78,873% | 73,123% | 0,198% | 1.278 |
| Shellcode | 50,407% | 86,713% | 63,753% | 0,051% | 143 |
| Worms | 37,5% | 17,647% | 24% | 0,002% | 17 |

**Chi tiết từng lớp, `stage2_nf_unsw_nb15_v3`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 33,438% | 75,352% | 46,32% | 0,09% | 142 |
| Backdoor | 71,429% | 0,406% | 0,808% | 0,001% | 1.231 |
| Benign | 99,996% | 99,948% | 99,972% | 0,045% | 216.447 |
| DoS | 20,813% | 22,563% | 21,653% | 0,273% | 749 |
| Exploits | 65,714% | 68,104% | 66,888% | 1,006% | 6.515 |
| Fuzzers | 63,651% | 83,461% | 72,222% | 1,107% | 5.369 |
| Generic | 88,013% | 78,257% | 82,849% | 0,139% | 3.040 |
| Reconnaissance | 74,864% | 59,761% | 66,465% | 0,237% | 2.761 |
| Shellcode | 60,248% | 72,388% | 65,763% | 0,054% | 268 |
| Worms | 28,205% | 52,381% | 36,667% | 0,012% | 21 |

**Chi tiết từng lớp, `stage2_unsw_v3_baseline`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 37,321% | 54,93% | 44,444% | 0,055% | 142 |
| Backdoor | 33,333% | 0,406% | 0,803% | 0,004% | 1.231 |
| Benign | 99,998% | 99,988% | 99,993% | 0,02% | 216.447 |
| DoS | 23,917% | 16,956% | 19,844% | 0,171% | 749 |
| Exploits | 63,962% | 70,913% | 67,259% | 1,132% | 6.515 |
| Fuzzers | 62,926% | 76,979% | 69,247% | 1,053% | 5.369 |
| Generic | 86,546% | 78,717% | 82,446% | 0,159% | 3.040 |
| Reconnaissance | 70,449% | 62,514% | 66,244% | 0,31% | 2.761 |
| Shellcode | 61,022% | 71,269% | 65,749% | 0,052% | 268 |
| Worms | 24,444% | 52,381% | 33,333% | 0,014% | 21 |

**Chi tiết từng lớp, `stage2_unsw_v3_no_iat`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 46,497% | 51,408% | 48,829% | 0,036% | 142 |
| Backdoor | 57,143% | 0,325% | 0,646% | 0,001% | 1.231 |
| Benign | 99,998% | 99,947% | 99,973% | 0,025% | 216.447 |
| DoS | 18,483% | 17,891% | 18,182% | 0,251% | 749 |
| Exploits | 66,577% | 68,611% | 67,579% | 0,976% | 6.515 |
| Fuzzers | 62,54% | 79,233% | 69,904% | 1,102% | 5.369 |
| Generic | 85,304% | 78,092% | 81,539% | 0,175% | 3.040 |
| Reconnaissance | 63,827% | 61,862% | 62,829% | 0,414% | 2.761 |
| Shellcode | 61,905% | 72,761% | 66,895% | 0,051% | 268 |
| Worms | 26,923% | 33,333% | 29,787% | 0,008% | 21 |

**Chi tiết từng lớp, `stage2_unsw_v3_no_propagation`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 39,013% | 61,268% | 47,671% | 0,058% | 142 |
| Backdoor | 40% | 0,325% | 0,645% | 0,003% | 1.231 |
| Benign | 99,998% | 99,978% | 99,988% | 0,02% | 216.447 |
| DoS | 16,527% | 15,754% | 16,131% | 0,253% | 749 |
| Exploits | 62,788% | 70,652% | 66,489% | 1,186% | 6.515 |
| Fuzzers | 61,875% | 77,202% | 68,694% | 1,105% | 5.369 |
| Generic | 87,147% | 77,171% | 81,856% | 0,148% | 3.040 |
| Reconnaissance | 71,707% | 54,618% | 62,007% | 0,255% | 2.761 |
| Shellcode | 60,299% | 75,373% | 66,998% | 0,056% | 268 |
| Worms | 28,125% | 42,857% | 33,962% | 0,01% | 21 |

**Chi tiết từng lớp, `stage2_unsw_v3_random_nodes`**

| Lớp | Precision | Recall | F1 | FAR | Support |
|---|---:|---:|---:|---:|---:|
| Analysis | 19,876% | 45,07% | 27,586% | 0,109% | 142 |
| Backdoor | 62,5% | 0,406% | 0,807% | 0,001% | 1.231 |
| Benign | 99,995% | 99,987% | 99,991% | 0,055% | 216.447 |
| DoS | 18,97% | 18,692% | 18,83% | 0,254% | 749 |
| Exploits | 64,247% | 69,977% | 66,99% | 1,103% | 6.515 |
| Fuzzers | 63,623% | 75,023% | 68,855% | 0,996% | 5.369 |
| Generic | 83,224% | 79,145% | 81,133% | 0,208% | 3.040 |
| Reconnaissance | 70,855% | 58,204% | 63,909% | 0,283% | 2.761 |
| Shellcode | 44,965% | 71,642% | 55,252% | 0,099% | 268 |
| Worms | 31,25% | 47,619% | 37,736% | 0,009% | 21 |

### Điểm làm việc

Ngưỡng luôn chọn trên tập validation rồi mới áp lên tập test. Điểm `far_0.0009` ứng với mức FAR 0,09% mà TE-G-SAGE công bố.

| Cấu hình | Điểm | Threshold | Precision | Recall | F1 | FAR | Đạt ràng buộc |
|---|---|---:|---:|---:|---:|---:|---|
| `stage1_nf_cse_cic_ids2018_v3` | f1 | 1.02035 | 21,663% | 26,887% | 23,994% | 14,4413% | — |
| `stage1_nf_cse_cic_ids2018_v3` | far_0.0009 | 14.00945 | 22,857% | 0,157% | 0,312% | 0,0787% | có |
| `stage1_nf_cse_cic_ids2018_v3` | far_0.01 | 4.25404 | 17,255% | 1,311% | 2,437% | 0,9337% | có |
| `stage1_nf_unsw_nb15_v2` | f1 | 0.45994 | 71,602% | 63,115% | 67,092% | 1,0369% | — |
| `stage1_nf_unsw_nb15_v2` | far_0.0009 | 3.33280 | 89,478% | 20,572% | 33,453% | 0,1002% | có |
| `stage1_nf_unsw_nb15_v2` | far_0.01 | 0.51612 | 73,713% | 60,696% | 66,574% | 0,8966% | có |
| `stage1_nf_unsw_nb15_v3` | f1 | 0.77139 | 96,467% | 98,791% | 97,615% | 0,3359% | — |
| `stage1_nf_unsw_nb15_v3` | far_0.0009 | 2.10803 | 98,654% | 88,635% | 93,376% | 0,1123% | có |
| `stage1_nf_unsw_nb15_v3` | far_0.01 | 0.36343 | 90,444% | 99,851% | 94,915% | 0,9795% | có |
| `stage1_unsw_v3_baseline` | f1 | 0.67385 | 96,273% | 98,726% | 97,484% | 0,3548% | — |
| `stage1_unsw_v3_baseline` | far_0.0009 | 2.05828 | 98,604% | 83,27% | 90,291% | 0,1095% | có |
| `stage1_unsw_v3_baseline` | far_0.01 | 0.34512 | 90,481% | 99,851% | 94,935% | 0,9753% | có |
| `stage1_unsw_v3_no_iat` | f1 | 0.89579 | 96,409% | 98,338% | 97,364% | 0,34% | — |
| `stage1_unsw_v3_no_iat` | far_0.0009 | 3.31698 | 98,977% | 74,607% | 85,081% | 0,0716% | có |
| `stage1_unsw_v3_no_iat` | far_0.01 | 0.41868 | 90,607% | 99,791% | 94,977% | 0,9605% | có |
| `stage1_unsw_v3_no_propagation` | f1 | 1.19327 | 94,324% | 98,821% | 96,52% | 0,5521% | — |
| `stage1_unsw_v3_no_propagation` | far_0.0009 | 4.39456 | 98,576% | 66,814% | 79,645% | 0,0896% | có |
| `stage1_unsw_v3_no_propagation` | far_0.01 | 0.57207 | 90,48% | 99,975% | 94,991% | 0,9767% | có |
| `stage1_unsw_v3_random_nodes` | f1 | 0.35290 | 96,022% | 98,139% | 97,069% | 0,3775% | — |
| `stage1_unsw_v3_random_nodes` | far_0.0009 | 1.11536 | 99,204% | 84,35% | 91,176% | 0,0628% | có |
| `stage1_unsw_v3_random_nodes` | far_0.01 | 0.23099 | 89,996% | 99,597% | 94,553% | 1,028% | có |
| `stage2_nf_cse_cic_ids2018_v3` | argmax | — | 99,716% | 81,374% | 89,616% | 0,0345% | — |
| `stage2_nf_cse_cic_ids2018_v3` | f1 | 0.10911 | 99,007% | 92,884% | 95,848% | 0,1384% | — |
| `stage2_nf_cse_cic_ids2018_v3` | far_0.0009 | 0.19077 | 99,468% | 91,014% | 95,053% | 0,0723% | có |
| `stage2_nf_cse_cic_ids2018_v3` | far_0.01 | 0.02518 | 94,386% | 94,037% | 94,211% | 0,8308% | có |
| `stage2_nf_unsw_nb15_v2` | argmax | — | 89,809% | 98,896% | 94,134% | 0,4649% | — |
| `stage2_nf_unsw_nb15_v2` | f1 | 0.92692 | 92,337% | 96,571% | 94,407% | 0,332% | — |
| `stage2_nf_unsw_nb15_v2` | far_0.0009 | 0.99211 | 97,663% | 69,889% | 81,474% | 0,0693% | có |
| `stage2_nf_unsw_nb15_v2` | far_0.01 | 0.00017 | 85,365% | 100% | 92,105% | 0,7102% | có |
| `stage2_nf_unsw_nb15_v3` | argmax | — | 99,446% | 99,955% | 99,7% | 0,0517% | — |
| `stage2_nf_unsw_nb15_v3` | f1 | 0.00000 | 98,917% | 100% | 99,456% | 0,1016% | — |
| `stage2_nf_unsw_nb15_v3` | far_0.0009 | 0.99862 | 99,497% | 98,442% | 98,967% | 0,0462% | có |
| `stage2_nf_unsw_nb15_v3` | far_0.01 | 0.00000 | 93,474% | 100% | 96,627% | 0,6482% | có |
| `stage2_unsw_v3_baseline` | argmax | — | 99,866% | 99,98% | 99,923% | 0,0125% | — |
| `stage2_unsw_v3_baseline` | f1 | 0.00000 | 99,269% | 100% | 99,633% | 0,0684% | — |
| `stage2_unsw_v3_baseline` | far_0.0009 | 0.99905 | 99,914% | 98,507% | 99,206% | 0,0079% | có |
| `stage2_unsw_v3_baseline` | far_0.01 | 0.00000 | 92,378% | 100% | 96,038% | 0,766% | có |
| `stage2_unsw_v3_no_iat` | argmax | — | 99,436% | 99,975% | 99,705% | 0,0527% | — |
| `stage2_unsw_v3_no_iat` | f1 | 0.00001 | 98,83% | 100% | 99,411% | 0,11% | — |
| `stage2_unsw_v3_no_iat` | far_0.0009 | 0.99836 | 99,485% | 97,96% | 98,716% | 0,0471% | có |
| `stage2_unsw_v3_no_iat` | far_0.01 | 0.00000 | 91,604% | 100% | 95,618% | 0,851% | có |
| `stage2_unsw_v3_no_propagation` | argmax | — | 99,767% | 99,98% | 99,873% | 0,0217% | — |
| `stage2_unsw_v3_no_propagation` | f1 | 0.00639 | 98,66% | 100% | 99,325% | 0,1261% | — |
| `stage2_unsw_v3_no_propagation` | far_0.0009 | 0.99477 | 99,905% | 98,965% | 99,433% | 0,0088% | có |
| `stage2_unsw_v3_no_propagation` | far_0.01 | 0.00000 | 91,35% | 100% | 95,479% | 0,8792% | có |
| `stage2_unsw_v3_random_nodes` | argmax | — | 99,861% | 99,945% | 99,903% | 0,0129% | — |
| `stage2_unsw_v3_random_nodes` | f1 | 0.00027 | 98,732% | 100% | 99,362% | 0,1192% | — |
| `stage2_unsw_v3_random_nodes` | far_0.0009 | 0.99783 | 99,919% | 98,403% | 99,155% | 0,0074% | có |
| `stage2_unsw_v3_random_nodes` | far_0.01 | 0.00002 | 90,821% | 100% | 95,19% | 0,9383% | có |

### Trọng số lớp đã dùng

- `stage2_nf_cse_cic_ids2018_v3`: Benign=0.022, Bot=0.2018, Brute_Force_-Web=2.288, Brute_Force_-XSS=4.2525, DDOS_attack-HOIC=0.0905, DDOS_attack-LOIC-UDP=1.5671, DDoS_attacks-LOIC-HTTP=0.1712, DoS_attacks-GoldenEye=0.3715, DoS_attacks-Hulk=0.2907, DoS_attacks-SlowHTTPTest=0.2831, DoS_attacks-Slowloris=0.4845, FTP-BruteForce=0.1479, Infilteration=0.212, SQL_Injection=4.4054, SSH-Bruteforce=0.2118
- `stage2_nf_unsw_nb15_v2`: Analysis=1.0995, Backdoor=1.132, Benign=0.0348, DoS=0.6926, Exploits=0.2968, Fuzzers=0.3529, Generic=0.4096, Reconnaissance=0.4663, Shellcode=1.3959, Worms=4.1196
- `stage2_nf_unsw_nb15_v3`: Analysis=1.6314, Backdoor=0.7874, Benign=0.034, DoS=0.714, Exploits=0.2699, Fuzzers=0.2978, Generic=0.4028, Reconnaissance=0.431, Shellcode=1.0793, Worms=4.3525
- `stage2_unsw_v3_baseline`: Analysis=1.6314, Backdoor=0.7874, Benign=0.034, DoS=0.714, Exploits=0.2699, Fuzzers=0.2978, Generic=0.4028, Reconnaissance=0.431, Shellcode=1.0793, Worms=4.3525
- `stage2_unsw_v3_no_iat`: Analysis=1.6314, Backdoor=0.7874, Benign=0.034, DoS=0.714, Exploits=0.2699, Fuzzers=0.2978, Generic=0.4028, Reconnaissance=0.431, Shellcode=1.0793, Worms=4.3525
- `stage2_unsw_v3_no_propagation`: Analysis=1.6314, Backdoor=0.7874, Benign=0.034, DoS=0.714, Exploits=0.2699, Fuzzers=0.2978, Generic=0.4028, Reconnaissance=0.431, Shellcode=1.0793, Worms=4.3525
- `stage2_unsw_v3_random_nodes`: Analysis=1.6314, Backdoor=0.7874, Benign=0.034, DoS=0.714, Exploits=0.2699, Fuzzers=0.2978, Generic=0.4028, Reconnaissance=0.431, Shellcode=1.0793, Worms=4.3525

---

# Các phép đo bổ trợ

Những con số dưới đây không sinh ra từ notebook huấn luyện. Mục 1 và 2 đọc từ `models/supplementary/supplementary_measurements.json`, tệp do `scripts/analysis/supplementary_measurements.py` sinh ra khi chạy trên CPU với dữ liệu gốc. Mục 3 đọc từ bảng tổng hợp thí nghiệm cắt bỏ. Mục 4 là số liệu trích từ các bài báo. Mỗi mục ghi điều kiện đo ngay đầu mục.

## 1. Lối tắt trong dữ liệu NF-UNSW-NB15-v3

**Điều kiện đo.** Toàn bộ 2.365.424 luồng, không chia tập, không huấn luyện gì. TTL dùng giá trị thô chưa chuẩn hoá, ROC-AUC lấy chính giá trị MAX_TTL làm điểm của lớp tấn công. Báo cáo dùng ở mục 4.9.1.

| Phép đo | Kết quả |
|---|---|
| Số luồng | 2.365.424 |
| Địa chỉ nguồn chỉ sinh lưu lượng tấn công | 4 địa chỉ: `175.45.176.0`, `175.45.176.1`, `175.45.176.2`, `175.45.176.3` |
| Địa chỉ nguồn chỉ sinh lưu lượng Benign | 36 |
| Địa chỉ nguồn sinh cả hai loại | **0** |
| `MAX_TTL` phổ biến nhất, lớp Benign | 32 |
| `MAX_TTL` phổ biến nhất, lớp tấn công | 255 |
| ROC-AUC khi dùng riêng `MAX_TTL` | 0,9984 |

## 2. Mốc tham chiếu dạng bảng, không dùng đồ thị

**Điều kiện đo.** Giao thức TE-G-SAGE của dự án (split_te_g_sage): sắp theo FLOW_START_MILLISECONDS, chia liên tục 60/30/10, huấn luyện trên tập train, đo trên tập test. Thuộc tính qua NetFlowPreprocessor khớp trên tập train, không có địa chỉ IP và mốc thời gian. Nhãn là cột Label hai lớp. Tập validation không dùng. Ngưỡng mặc định của predict(). Bộ thuộc tính gồm 49 cột, scikit-learn 1.9.1. Báo cáo dùng ở bảng 4.24, làm tròn hai chữ số thập phân cho F1 và PR-AUC, bốn chữ số cho FAR.

| Tập | Số luồng | Số luồng tấn công |
|---|---:|---:|
| train | 1.419.254 | 42.459 |
| val | 709.628 | 65.138 |
| test | 236.542 | 20.096 |

| Mô hình | Cấu hình | F1 lớp tấn công | FAR | PR-AUC |
|---|---|---:|---:|---:|
| Cây quyết định sâu 3, đủ thuộc tính | `DecisionTreeClassifier(max_depth=3, random_state=42)` | 99,794% | 0,0333% | 99,797% |
| Cây quyết định sâu 3, bỏ `MIN_TTL` và `MAX_TTL` | `DecisionTreeClassifier(max_depth=3, random_state=42), bỏ MIN_TTL và MAX_TTL` | 99,551% | 0,0661% | 99,137% |
| Hồi quy logistic, đủ thuộc tính | `LogisticRegression(max_iter=1000), tham số còn lại mặc định` | 99,566% | 0,018% | 99,769% |

Thời gian chạy script: 28,6 giây.

## 3. Thí nghiệm cắt bỏ kiến trúc

**Điều kiện đo.** Bốn cấu hình huấn luyện lại từ đầu qua cả hai giai đoạn trên NF-UNSW-NB15-v3, cùng một phiên, cùng hạt giống 42, giao thức `chronological_80_10_10`. Macro F1 đo trên bài toán nhiều lớp. Recall, F1 và FAR của lớp tấn công đo trên bài toán hai lớp quy đổi. Mỗi cấu hình chạy **một lần**, nên chênh lệch dưới 1 điểm chưa tách khỏi dao động giữa các lần chạy. Nguồn: `models/saved/twoDTS_ablation/08_tong_hop/tables/ablation_summary.csv`. Báo cáo dùng ở mục 4.6.

| Cấu hình | Chiều cạnh | Macro F1 | Chênh lệch Macro F1 | Recall tấn công | F1 tấn công | FAR |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 49 | 54,936% | 0,000 điểm | 99,98% | 99,923% | 0,012% |
| `random_nodes` | 49 | 52,109% | -2,827 điểm | 99,945% | 99,903% | 0,013% |
| `no_iat` | 41 | 54,616% | -0,320 điểm | 99,975% | 99,705% | 0,053% |
| `no_propagation` | 49 | 54,444% | -0,492 điểm | 99,98% | 99,873% | 0,022% |

Phép triệt tiêu từng thuộc tính ở mục 4.7.1 là công cụ giải thích, không phải thí nghiệm cắt bỏ kiến trúc.

## 4. Số liệu do các nghiên cứu đối sánh công bố

**Điều kiện đo.** Trích nguyên từ bản gốc trong `references/`, không đo lại. Số hiệu tài liệu theo danh mục của báo cáo.

| Nguồn | Bộ dữ liệu | Bài toán | Chỉ số | Điều kiện đo của bài báo |
|---|---|---|---|---|
| GraphIDS [10], Bảng 3 | NF-UNSW-NB15-v3 | hai lớp | Macro F1 99,61% ± 0,84% · PR-AUC 99,98% ± 0,07% | phân tầng ngẫu nhiên 80/10/10, trung bình nhiều hạt giống |
| GraphIDS [10], Bảng 3 | NF-UNSW-NB15-v2 | hai lớp | Macro F1 92,64% ± 2,17% · PR-AUC 81,16% ± 3,67% | như trên |
| GraphIDS [10], Bảng 3 | NF-CSE-CIC-IDS2018-v3 | hai lớp | Macro F1 94,47% ± 2,13% · PR-AUC 88,19% ± 3,47% | như trên, toàn bộ dữ liệu |
| Anomal-E, do GraphIDS [10] đo lại, Bảng 3 | NF-UNSW-NB15-v3 | hai lớp | Macro F1 94,59% ± 0,09% · PR-AUC 90,32% ± 0,41% | như GraphIDS, biến thể tốt nhất theo PR-AUC |
| Anomal-E, do GraphIDS [10] đo lại, Bảng 3 | NF-UNSW-NB15-v2 | hai lớp | Macro F1 91,56% ± 2,17% · PR-AUC 74,89% ± 0,74% | như trên |
| Anomal-E, do GraphIDS [10] đo lại, Bảng 3 | NF-CSE-CIC-IDS2018-v3 | hai lớp | Macro F1 67,09% ± 3,94% · PR-AUC 25,55% ± 3,83% | như trên, toàn bộ dữ liệu |
| TE-G-SAGE [14] | NF-UNSW-NB15-v3 | hai lớp | P 99,06% · R 99,99% · F1 99,52% · FAR 0,09% | chia theo thời gian 60/30/10 |
| TE-G-SAGE [14], Bảng 7 | NF-UNSW-NB15-v3 | nhiều lớp | Acc 95,586% · P 49,419% · R 62,738% · Macro F1 49,057% · FAR 0,446% | chia theo thời gian 60/30/10 |
| GCN, trong TE-G-SAGE [14], Bảng 7 | NF-UNSW-NB15-v3 | nhiều lớp | Acc 97,257% · P 39,587% · R 39,477% · Macro F1 38,781% · FAR 0,301% | như trên |
| XGBoost, trong TE-G-SAGE [14], Bảng 7 | NF-UNSW-NB15-v3 | nhiều lớp | Acc 97,338% · P 68,957% · R 55,742% · Macro F1 56,782% · FAR 0,273% | như trên |
| Anomal-E-IF, bài gốc [4], Bảng 4 và 8 | NF-UNSW-NB15-v2 | hai lớp | Acc 98,66% · Macro F1 92,35% · DR 98,77% | giao thức riêng của Anomal-E, 4% nhiễm tấn công trong tập huấn luyện, báo cáo không dùng |

TE-G-SAGE theo lớp, bài toán nhiều lớp, F1: Analysis 0,305 · Backdoor 0,071 · Benign 1,000 · DoS 0,260 · Exploits 0,613 · Fuzzers 0,501 · Generic 0,796 · Reconnaissance 0,579 · Shellcode 0,221 · Worms 0,500.

TCG-IDS chưa có bản gốc nên **không trích số nào**.
