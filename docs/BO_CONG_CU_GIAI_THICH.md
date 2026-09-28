# Bộ công cụ giải thích: ba công cụ, ba câu hỏi

Đồ án dùng ba công cụ giải thích, đặt trong `src/tgn_nids/explainers/`. Chúng
**bổ sung cho nhau**, không thay thế nhau, vì mỗi công cụ trả lời một câu hỏi
khác và đo trên một nhánh khác của mô hình.

| Công cụ | Câu hỏi nó trả lời | Nhánh nó đo | Cách làm |
|---|---|---|---|
| `TGNExplainer` | Cột NetFlow nào đẩy quyết định | đặc trưng luồng | triệt tiêu từng cột, một lượt chạy mỗi cột |
| `GNNExplainer` | Cạnh lân cận nào đẩy quyết định | cấu trúc lân cận | tối ưu mặt nạ bằng đạo hàm |
| `SubgraphExtractor` | Cảnh báo nằm ở đâu trong mạng | cấu trúc lân cận | trích k bước, không đo trọng số |

Đề cương nêu đích danh `GNNExplainer` cùng với subgraph k bước, và cả ba công cụ
trên đều có mặt trong mã.

---

## 1. `TGNExplainer`: đặc trưng nào quan trọng

Phép triệt tiêu từng cột không cần tối ưu nên kết quả **tất định**: chạy hai lần
cho hai kết quả giống hệt. Chi phí cố định và biết trước, một lượt chạy cho mỗi
cột, tức 49 lượt cho bộ đặc trưng bản v3. Không có siêu tham số nào phải tinh
chỉnh.

Với câu hỏi "đặc trưng nào quan trọng", đây là công cụ nên dùng.

---

## 2. `GNNExplainer`: cạnh nào quan trọng

Đây là công cụ duy nhất trong ba công cụ đo được **trọng số của từng cạnh**. Nhờ
nó, hình subgraph vẽ được bề rộng cạnh theo mức đóng góp vào quyết định thay vì
theo số luồng, tức đúng tinh thần khoanh vùng nguyên nhân mà đề cương mô tả.

Điều kiện để đạo hàm chảy được về từng cạnh là **khối nhúng phải nằm trong vòng
tối ưu**, để mặt nạ cạnh ảnh hưởng tới véc-tơ đỉnh:

```python
edge_attr_masked = edge_attr * edge_mask.unsqueeze(1)
z = self.embedding(node_memory, edge_index, edge_attr_masked)
p_attack = self._attack_probability(self.detector(z[src], z[dst], attr_target))
```

### 2.1. Kết quả đo được

Nguồn:
`models/saved/twoDTS_phan_tich/tables/08_trong_so_canh_gnnexplainer/trong_so_canh_giai_thich.csv`,
đo trên 600 cạnh lân cận của một luồng.

| Đại lượng | Giá trị |
|---|---:|
| Độ lệch chuẩn trọng số | 35,360% |
| Trọng số nhỏ nhất | 10,817% |
| Trọng số lớn nhất | 89,778% |
| Trung vị | 12,249% |
| Số cạnh vượt ngưỡng 0,5 | 189 trên 600 |

Phần lớn cạnh bị hạ hẳn xuống, một nhóm nhỏ được giữ lên cao. Đó đúng là dạng
phân bố mà một phép giải thích theo cạnh phải cho ra.

### 2.2. Không đụng tới đường huấn luyện

Ba tệp liên quan là `explainers/gnn_explainer.py`, `explainers/__init__.py` và
notebook phân tích. `runner.py` không nhập tệp nào trong ba tệp đó, và hai
notebook huấn luyện cũng không nhắc tới chúng. Mô hình đã lưu dùng nguyên, không
huấn luyện lại.

Notebook phân tích có ô tự kiểm: nó chụp `state_dict` của bộ phân loại trước khi
giải thích, so lại sau đó, và **dừng với lỗi** nếu có tham số nào bị đổi. Phép
chạy trên mô hình thật cho 0 trên 11 tham số của bộ phân loại bị đổi và 0 trên 11
của khối nhúng.

---

## 3. Mặt nạ không làm cảnh báo biến mất

Nguồn: cùng tệp trọng số cạnh ở mục 2.1. Trong 600 cạnh, 411 cạnh có trọng số
không vượt 0,5. Chúng không bị xoá khỏi đồ thị. GNNExplainer áp mặt nạ mềm lên
tất cả các cạnh, rồi xác suất tấn công của cảnh báo giảm từ **99,989%** xuống
**98,647%**.

Một cảnh báo không đủ để suy ra độ bền chung của phép tổng hợp lân cận, cũng
không định lượng được đóng góp của đồ thị cho toàn mô hình. Câu hỏi đó được đo
riêng bằng thí nghiệm cắt bỏ kiến trúc tại
`docs/THUC_NGHIEM_CAT_BO_KIEN_TRUC.md`; phép đo này chỉ có một hạt giống.

---

## 4. Hình subgraph cho báo cáo

`plot_ego_network` nhận tham số `edge_weights`. Không truyền thì bề rộng cạnh
theo số luồng; có truyền thì bề rộng theo trọng số giải thích.

Mục 7 và mục 8 của notebook phân tích vì vậy sinh hai hình cùng cấu trúc nhưng
mã hoá hai đại lượng khác nhau. Đặt cạnh nhau trong báo cáo thì khác biệt hiện
ra rõ.

---

## 5. Ba hạn chế phải nêu khi trình bày kết quả giải thích

**Kết quả của `GNNExplainer` phụ thuộc hạt giống.** Mặt nạ khởi tạo ngẫu nhiên
rồi mới tối ưu. Notebook cố định `GNNEXPLAINER_SEED = 7` để tái hiện cùng phép
đo.

**Đây chỉ là một ví dụ cục bộ.** Báo cáo chọn cảnh báo có xác suất tấn công cao
nhất trong mẫu ngẫu nhiên 600 luồng với hạt giống 42. Hình thu được không đại
diện cho mọi cảnh báo hoặc mọi kiểu tấn công.

**Một cạnh có trọng số cao không phải nguyên nhân duy nhất.** Mặt nạ thể hiện
phần đồ thị giúp giữ lại dự đoán của mô hình trong phép tối ưu này. Nó không tự
nó chứng minh quan hệ nhân quả.
