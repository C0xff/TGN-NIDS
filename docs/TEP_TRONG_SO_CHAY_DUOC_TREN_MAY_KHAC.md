# Tệp trọng số mở được trên máy khác

Tệp trọng số `.pt` của đồ án mở được trên bất kỳ máy nào có PyTorch, **không cần
cài mã nguồn dự án** và **không cần tắt chốt an toàn** của PyTorch. Tài liệu này
giải thích điều kiện để làm được vậy, viết cho người chưa quen chi tiết bên trong
PyTorch.

---

## 1. `torch.load` thật ra làm gì

Khi lưu mô hình, PyTorch dùng cơ chế **pickle** của Python. Pickle ghi lại cả
**cách dựng lại một đối tượng Python**, nên khi mở tệp nó sẽ nhập gói, gọi hàm và
truyền tham số theo đúng hướng dẫn ghi trong tệp.

Đó là chỗ nguy hiểm. Một tệp `.pt` do người khác gửi có thể chứa hướng dẫn làm
bất cứ việc gì: đọc tệp trên máy, gửi dữ liệu ra ngoài, xoá thư mục. **Mở tệp là
chạy mã trong tệp.** Người mở không thấy gì bất thường, vì tệp vẫn nạp ra một mô
hình hoạt động bình thường.

Vì lý do đó, từ phiên bản 2.6, PyTorch đặt mặc định của tham số `weights_only` là
`True`:

| Giá trị | Nghĩa là gì |
|---|---|
| `weights_only=True` | Chỉ khôi phục **tensor, số, chuỗi, danh sách, từ điển**. Gặp bất kỳ đối tượng của lớp nào khác thì từ chối, không chạy gì cả. |
| `weights_only=False` | Khôi phục mọi thứ, kể cả đối tượng của lớp bất kỳ. Chạy mã trong tệp. |

Toàn bộ chỗ nạp checkpoint trong dự án dùng `weights_only=True`.

---

## 2. Vì sao tệp trọng số buộc phải chứa bộ tiền xử lý

Ngoài trọng số, tệp của đồ án bắt buộc chứa **bộ tiền xử lý**.

Lý do: trước khi đưa vào mô hình, mỗi cột đặc trưng được **chuẩn hoá**, tức trừ
đi trung bình rồi chia cho độ lệch chuẩn của chính cột đó, đo trên tập huấn
luyện. Ví dụ cột `IN_BYTES` có trung bình 4.000 và độ lệch chuẩn 12.000; một
luồng 16.000 byte đi vào mô hình dưới dạng số 1,0.

Nếu lúc suy luận dùng trung bình và độ lệch chuẩn khác, cùng một luồng sẽ thành
một con số khác, và mô hình cho ra dự đoán khác. **Nguy hiểm ở chỗ chương trình
không báo lỗi**: nó vẫn chạy, vẫn trả về kết quả, chỉ là kết quả sai. Vì vậy các
tham số chuẩn hoá phải đi kèm trọng số trong cùng một tệp.

---

## 3. Nguyên tắc: cất dữ liệu, không cất đối tượng

Một tệp cất nguyên **đối tượng** `NetFlowPreprocessor` sẽ ràng tệp vào hai điều
kiện: phải đặt `weights_only=False` để mở, và phải có sẵn gói `tgn_nids` trên
đường dẫn tìm kiếm của Python, vì pickle cần đúng lớp đó để dựng lại đối tượng.
Thiếu gói thì tệp báo `ModuleNotFoundError: No module named 'tgn_nids'` và không
mở ra được ở bất kỳ đâu khác. Điều kiện thứ hai là điều đáng nói khi nộp bài:
người chấm nhận được tệp trọng số nhưng nếu chưa cài đúng cây thư mục mã nguồn
thì không mở nổi.

Bên trong `NetFlowPreprocessor` không có gì cần tới một lớp Python để tồn tại.
Liệt kê thuộc tính của nó cho thấy: vài danh sách tên cột, vài cờ đúng sai, và
hai đối tượng của thư viện scikit-learn. Hai đối tượng đó, mở ra xem, cũng chỉ
chứa mảng số:

| Đối tượng | Chứa gì |
|---|---|
| `StandardScaler` | `mean_`, `var_`, `scale_` — ba mảng, mỗi mảng 41 số, một số cho mỗi cột |
| `LabelEncoder` | `classes_` — mảng 10 tên lớp |

Nên tệp cất **bản kê các con số đó**, rồi lúc mở thì dựng lại đối tượng từ bản
kê. Hai phương thức của `NetFlowPreprocessor` làm việc này:

- `state_dict()` trả về một từ điển chỉ gồm số, chuỗi, danh sách, từ điển.
- `from_state_dict(state)` nhận từ điển đó và dựng lại một bộ tiền xử lý.

`runner.py` lưu `"preprocessor_state": preprocessor.state_dict()`, và mọi chỗ nạp
gọi `from_state_dict` sau khi đọc bằng `weights_only=True`.

### 3.1. Chế độ an toàn chặt hơn vẻ ngoài của nó

`weights_only=True` từ chối cả `numpy.str_`, không riêng đối tượng của các lớp
tự định nghĩa. Danh sách `class_names` vì vậy phải đổi tường minh sang `str` của
Python:

```python
class_names = [str(name) for name in label_encoder.classes_]
```

Thiếu một dòng như vậy thì tệp vẫn ghi ra bình thường và chỉ đổ vỡ lúc nạp. Đây
là lý do phần tự kiểm ở mục 4 phải là phép thử tự động chứ không phải đọc mã rồi
tin là đúng.

### 3.2. Bản kê mang số phiên bản

Bản kê có trường `format_version`. Gặp số lạ thì `from_state_dict` dừng với lỗi
rõ ràng, thay vì dựng ra một bộ tiền xử lý thiếu tham số rồi chạy tiếp và cho ra
dự đoán sai trong im lặng.

---

## 4. Cách tự kiểm

Ba phép thử trong `tests/test_runner_guards.py`, lớp `TestPreprocessorStateDict`:

- **Khứ hồi.** Khớp một bộ tiền xử lý, cất thành bản kê, dựng lại, rồi so kết quả
  `transform` của bản gốc với bản dựng lại. Phải **giống hệt từng phần tử**,
  không phải gần đúng. Đây là phép thử quan trọng nhất, vì một tham số chuẩn hoá
  sai không làm chương trình dừng mà chỉ làm dự đoán sai trong im lặng.
- **Nạp ở chế độ an toàn.** Ghi bản kê bằng `torch.save`, đọc lại bằng
  `torch.load(..., weights_only=True)`. Lệnh này chạy trót lọt chính là bằng
  chứng bản kê không chứa đối tượng nào của dự án.
- **Từ chối bản kê sai phiên bản.** Đưa vào một `format_version` lạ và đòi hàm
  dựng lại phải dừng với lỗi.

Muốn tự kiểm bằng tay, cách thuyết phục nhất là mở tệp ở một tiến trình **không
có mã nguồn dự án**:

```bash
cd /tmp
env -u PYTHONPATH python3 -c "
import torch
d = torch.load('<đường dẫn tới tệp .pt>', map_location='cpu', weights_only=True)
print(list(d))
"
```

Chạy được ở đó nghĩa là chạy được trên máy người chấm.

---

## 5. Hai hệ quả cần biết

**Checkpoint phải phù hợp với định dạng dữ liệu hiện tại.** Tệp chứa đối tượng
đã pickle bị mã hiện tại từ chối. Không chạy lại notebook đã công bố hoặc ghi đè
checkpoint để xử lý trường hợp này. Giữ nguyên tệp gốc; một phép chuyển đổi hay
kiểm tra tương thích chỉ nên thực hiện trong quy trình độc lập.

**Nguyên tắc rút ra, dùng được ở chỗ khác:** thứ cần lưu lâu dài nên lưu ở dạng
dữ liệu, không ở dạng đối tượng của một lớp cụ thể. Đối tượng ràng tệp vào phiên
bản mã đã tạo ra nó; dữ liệu thì không.
