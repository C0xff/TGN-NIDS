"""Phát lại từng dòng NetFlow để mô phỏng luồng dữ liệu thời gian thực."""

import os
import time
import pandas as pd


class LogGenerator:
    """Đọc dữ liệu mẫu và phát tuần tự từng bản ghi."""

    def __init__(self, file_path=None):
        if file_path is None:
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            file_path = os.path.join(project_root, "data", "samples",
                                     "demo_1_cung_phan_phoi.csv")

        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        """Nạp tệp CSV đã cấu hình vào bộ nhớ."""
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu giả lập tại: {self.file_path}")

    def stream_logs(self, delay_sec=1.0, loop=True):
        """Phát từng bản ghi theo nhịp đặt trước và tùy chọn lặp lại."""
        idx = 0
        total_rows = len(self.df)
        while True:
            row = self.df.iloc[idx].to_dict()
            yield row

            idx += 1
            if idx >= total_rows:
                if loop:
                    idx = 0
                else:
                    break

            if delay_sec > 0:
                time.sleep(delay_sec)

    def get_all_logs(self):
        """Trả về bản sao dữ liệu để xử lý theo lô."""
        return self.df.copy()


if __name__ == "__main__":
    print("Đang khởi động trình phát lại log NetFlow...")
    generator = LogGenerator()
    for log in generator.stream_logs(delay_sec=0.5, loop=False):
        src_info = f"{log.get('ipv4_src_addr', 'N/A')}:{log.get('L4_SRC_PORT', '')}"
        dst_info = f"{log.get('ipv4_dst_addr', 'N/A')}:{log.get('L4_DST_PORT', '')}"
        attack_info = log.get('Attack', 'Normal')
        print(f"[luồng] {src_info} -> {dst_info} | Nhãn: {attack_info}")
