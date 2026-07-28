
# Mã nguồn tham khảo
- https://github.com/curiousML/epsilon-fairness
- paper: Fairness Guarantees in Multi-class Classification with Demographic Parity - Journal of Machine Learning Research 25 (2024) 1-46

# Kết quả thực nghiệm
- Ứng với thực nghiệm trên synthetic data thì sẽ có folder tên tương ứng trong folder outputs
- Đối với dữ liệu thực tế thì chia ra chạy binary case và multi-class cũng có folder tên tương ứng

# Cách chạy

Toàn bộ lệnh dưới đây cần được chạy tại thư mục gốc của repository. Khuyến nghị
dùng Python 3.13, là phiên bản đã được kiểm tra với mã nguồn hiện tại.

## 1. Cài đặt trên Windows

Mở PowerShell tại thư mục repository và tạo môi trường ảo:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Nếu PowerShell chặn activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Nếu không muốn activate môi trường, có thể thay `python` trong các lệnh Windows
bằng `.\.venv\Scripts\python.exe`.

Cài toàn bộ dependency:

```powershell
python -m pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.0 matplotlib==3.11.1 seaborn==0.13.2 scikit-learn==1.9.0 lightgbm==4.7.0 fairlearn==0.14.0 aif360==0.6.1 tensorflow==2.21.0 statsmodels==0.14.6 cvxpy==1.9.2 osqp==1.1.3 tqdm==4.69.0 ipykernel==7.3.0 jupyter-client==8.9.1 notebook
```

Đăng ký kernel cho Jupyter:

```powershell
python -m ipykernel install --user --name ktdll-fairness --display-name "Python (KTDLL Fairness)"
```

## 2. Cài đặt trên Linux

Các lệnh sau áp dụng cho Bash. Máy cần có Python 3.13 và module `venv`. Trên
Ubuntu/Debian, nếu chưa có `venv`, cài bằng:

```bash
sudo apt update
sudo apt install python3.13 python3.13-venv
```

Tạo và kích hoạt môi trường:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Cài toàn bộ dependency:

```bash
python -m pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.0 matplotlib==3.11.1 seaborn==0.13.2 scikit-learn==1.9.0 lightgbm==4.7.0 fairlearn==0.14.0 aif360==0.6.1 tensorflow==2.21.0 statsmodels==0.14.6 cvxpy==1.9.2 osqp==1.1.3 tqdm==4.69.0 ipykernel==7.3.0 jupyter-client==8.9.1 notebook
```

Đăng ký kernel:

```bash
python -m ipykernel install --user --name ktdll-fairness --display-name "Python (KTDLL Fairness)"
```

## 3. Kiểm tra môi trường

Kiểm tra các package chính và solver của CVXPY:

```bash
python -c "import numpy, pandas, scipy, sklearn, lightgbm, fairlearn, tensorflow, cvxpy; print('Dependencies: OK'); print('CVXPY solvers:', cvxpy.installed_solvers())"
```

Chạy unit test trước khi thực hiện các thí nghiệm:

```bash
python -m unittest tests.test_reproduction_protocols tests.test_fair_projection_runtime -v
```

Các test phải kết thúc với trạng thái `OK`.

## 4. Mở và chạy notebook

Khởi động notebook:

```bash
python -m notebook experimentation_epsilon_fairness_update.ipynb (mở notebook từ terminal - optional)
```

Trong giao diện Jupyter:

1. Chọn kernel **Python (KTDLL Fairness)**.
2. Restart kernel để xóa trạng thái từ lần chạy trước.
3. Chọn **Run All Cells** để chạy toàn bộ notebook theo đúng thứ tự.

Notebook chứa các bước tải dữ liệu cục bộ, tiền xử lý, định nghĩa thuật toán
\(\epsilon\)-fair, chạy synthetic experiments, binary experiments và
multi-class experiments. Dữ liệu cần thiết đã nằm trong thư mục `datasets/`.

Các full experiment chạy 30 repetitions, đồng thời tuning RF/GBM và chạy nhiều
mức tolerance, nên có thể mất nhiều thời gian. Có thể chạy smoke test được định
nghĩa trong notebook trước, kiểm tra kết quả tại `outputs/multiclass_smoke/`, rồi
mới chạy cấu hình full.

## 5. Kết quả đầu ra

Các runner tự động lưu kết quả dưới thư mục `outputs/`:

| Thư mục | Nội dung |
|---|---|
| `outputs/synthetic/` | Synthetic Figures 1-4 và Figure 10 |
| `outputs/binary/` | DRUG/CRIME binary Figures 6-7 |
| `outputs/multiclass_smoke/` | Kết quả kiểm tra nhanh multi-class |
| `outputs/multiclass/` | Kết quả full multi-class |

Mỗi protocol lưu:

- file raw CSV cho từng repetition;
- file summary CSV chứa mean và standard deviation;
- file JSON ghi lại cấu hình;
- các biểu đồ được sinh bởi protocol tương ứng.

## 6. Solver tùy chọn cho Fair-transport

Fair-transport có thể chạy bằng các solver đi kèm CVXPY. Cấu hình `AUTO` ưu tiên:

- LP: `CBC`, sau đó `CLARABEL`, `SCIPY`, `SCS`;
- QP: `OSQP`, sau đó `CLARABEL`, `SCS`.

`CBC` không bắt buộc. Nếu muốn dùng đúng solver của repository Fair-transport
gốc, cài thêm:

```bash
python -m pip install cylp
```

Kiểm tra lại danh sách solver:

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

Nếu danh sách có `CBC`, chế độ `AUTO` sẽ tự động ưu tiên solver này.

## 7. Xử lý lỗi thường gặp

### Không import được package dù đã cài

Kiểm tra Python mà notebook đang sử dụng:

```python
import sys
print(sys.executable)
```

Đường dẫn phải trỏ tới `.venv`. Nếu không, chọn lại kernel **Python (KTDLL
Fairness)** và restart notebook.

### Windows báo không thể kích hoạt môi trường

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `No module named 'cvxpy'`

```bash
python -m pip install cvxpy==1.9.2 osqp==1.1.3
```

Sau khi cài, restart kernel trước khi chạy lại notebook.

### Không tìm thấy solver cho Fair-transport

```bash
python -m pip install osqp==1.1.3
```

Không bắt buộc phải có CBC vì adapter sẽ tự động sử dụng solver LP khác có sẵn.

### TensorFlow hiển thị thông báo oneDNN

Đây là thông báo về tối ưu floating-point, không phải lỗi. Sai khác số học rất
nhỏ giữa các hệ điều hành là bình thường; cần giữ cố định seed và version
dependency khi so sánh kết quả.

# FairProjection
- Mã nguồn tham khảo:
  - https://github.com/khanhchung101/ktdll-fairprojection - repo của nhóm reproduce 
  - https://github.com/uiuctml/fair-classification - repo của tác giả 
- Core vendored: `third_party/fair_projection/`
- Adapter dùng chung base model/split với notebook:
  `fairness_baselines/fair_projection_adapter.py`
- FairProjection cần thêm `tensorflow`, `cvxpy` và `tqdm`.
- Smoke test trong notebook chỉ dùng một ADMM iteration để kiểm tra wiring; không
  dùng kết quả smoke test trong báo cáo.
- Full multi-class run dùng Statistical Parity, cross-entropy,
  `alpha={0,0.1,0.2,0.5,0.75}`, `rho=2`, `max_iter=500`.
- Backend mặc định `method="tf"` cần TensorFlow eager mode. Runner kiểm tra điều
  kiện này trước khi chạy ADMM và báo lỗi có hướng xử lý nếu kernel đã bị chuyển
  sang graph mode.
- AIF360 AdversarialDebiasing chạy trong một Python subprocess riêng vì baseline
  này cần TensorFlow v1 graph mode. Do đó chạy binary experiment trước không còn
  vô hiệu hóa eager mode của FairProjection trong multi-class experiment.
- Có thể đặt `fair_projection_method="np"` để dùng fallback NumPy/CVXPY. Đường
  chạy này không import TensorFlow.

# Fair-transport
- Mã nguồn tham khảo:
  - https://github.com/uiuctml/fair-classification - repo của tác giả
- Baseline này là thuật toán Wasserstein-barycenter trong *Fair and Optimal
  Classification via Post-Processing* (Xian, Yin, Zhao, ICML 2023).
- Core: `third_party/fair_transport/postprocess.py`; adapter:
  `fairness_baselines/fair_transport_adapter.py`.
- Full multi-class run dùng chung base probabilities và calibration/test split,
  với `alpha={0,0.1,0.2,0.3,0.4}`.
- Chế độ `AUTO` ưu tiên solver giống repository ICML 2023: CBC qua `cylp`
  và OSQP; nếu CBC chưa có thì dùng LP solver sẵn có của CVXPY và ghi solver
  thực tế vào diagnostics.

# Protocol thực nghiệm lặp lại

- `reproduction_protocols.py` triển khai protocol 30 repetitions cho synthetic
  Figures 1-4, Figure 10 và real-data binary Figures 6-7.
- Mỗi runner lưu raw CSV, summary mean/std CSV và config JSON dưới `outputs/`.
- Fairlearn dùng grid paper `{0.0001,0.5,1,2.5,5,10}` cho tham số
  `ExponentiatedGradient.eps`; trường này được lưu riêng dưới tên
  `FairlearnTolerance`, không dùng làm `DemographicParity.difference_bound`.
- Chạy smoke test/unit test trước full run:
  `python -m unittest tests.test_reproduction_protocols -v`.

# Dataset
- https://www.kaggle.com/datasets/mexwell/drug-consumption-classification
- https://www.kaggle.com/datasets/chaditya95/communities-and-crime-data-set/data
