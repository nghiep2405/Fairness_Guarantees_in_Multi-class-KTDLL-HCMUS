# Hướng dẫn cài đặt và chạy đầy đủ repository


## 1. Tạo môi trường mới

Các lệnh dưới đây dành cho PowerShell tại thư mục gốc repository.

Khuyến nghị dùng Python 3.13, là phiên bản đã được kiểm tra với dependency hiện
tại của repository:

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

Có thể bỏ qua activation và luôn gọi trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

## 2. Cài toàn bộ thư viện

Lệnh cài đặt đầy đủ:

```powershell
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn lightgbm fairlearn aif360 tensorflow statsmodels cvxpy osqp tqdm ipykernel jupyter-client
```

Các phiên bản đã được kiểm tra thành công trong `.venv` ngày 23/07/2026:

```text
Python          3.13.0
numpy           2.5.1
pandas          3.0.5
scipy           1.18.0
matplotlib      3.11.1
seaborn         0.13.2
scikit-learn    1.9.0
lightgbm        4.7.0
fairlearn       0.14.0
aif360          0.6.1
tensorflow      2.21.0
statsmodels     0.14.6
cvxpy           1.9.2
osqp            1.1.3
tqdm            4.69.0
ipykernel       7.3.0
jupyter-client  8.9.1
```

Để tái tạo chính xác môi trường đã kiểm tra:

```powershell
python -m pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.0 matplotlib==3.11.1 seaborn==0.13.2 scikit-learn==1.9.0 lightgbm==4.7.0 fairlearn==0.14.0 aif360==0.6.1 tensorflow==2.21.0 statsmodels==0.14.6 cvxpy==1.9.2 osqp==1.1.3 tqdm==4.69.0 ipykernel==7.3.0 jupyter-client==8.9.1
```

## 3. Vai trò của từng nhóm thư viện

| Thành phần | Thư viện |
|---|---|
| Xử lý dữ liệu và tính toán | `numpy`, `pandas`, `scipy` |
| Mô hình cơ sở | `scikit-learn`, `lightgbm` |
| Vẽ biểu đồ | `matplotlib`, `seaborn` |
| Fairlearn baseline | `fairlearn` |
| Fair-adversarial baseline | `aif360`, `tensorflow` |
| FairProjection | `tensorflow`, `cvxpy`, `scipy`, `tqdm` |
| Fair-transport | `cvxpy`, `scikit-learn`, `osqp` |
| Notebook kernel | `ipykernel`, `jupyter-client` |
| Phụ trợ thống kê | `statsmodels` |

Không cần cài `torch`, `transformers` hoặc Hugging Face `datasets` để chạy
Fair-transport trong repository này. Các package đó xuất hiện trong repository
upstream nhưng không được phần Algorithm 2 đã tích hợp sử dụng.

## 4. Solver cho Fair-transport

Fair-transport dùng hai bài toán:

- LP Wasserstein-barycenter;
- QP để trích xuất score-shift map.

Cấu hình mặc định `AUTO`:

1. LP ưu tiên `CBC`, sau đó `CLARABEL`, `SCIPY`, `SCS`;
2. QP ưu tiên `OSQP`, sau đó `CLARABEL`, `SCS`.

Môi trường đã kiểm tra chạy thành công với:

```text
LP: CLARABEL
QP: OSQP
```

Repository ICML 2023 gốc dùng CBC. Nếu cần bám đúng solver upstream:

```powershell
python -m pip install cylp
```

Sau đó kiểm tra:

```python
import cvxpy as cp
print(cp.installed_solvers())
```

Nếu danh sách có `CBC`, chế độ `AUTO` sẽ tự ưu tiên CBC. Solver thực tế luôn
được lưu trong các cột `TransportMIPSolver` và `TransportQPSolver`.

## 5. Đăng ký Jupyter kernel

Sau khi cài dependency:

```powershell
python -m ipykernel install --user --name ktdll-fairness --display-name "Python (KTDLL Fairness)"
```

Đóng và mở lại notebook, sau đó chọn kernel **Python (KTDLL Fairness)**.

Nếu kernel cũ vẫn xuất hiện, kiểm tra:

```powershell
python -m jupyter_client.kernelspecapp list
```

## 6. Kiểm tra dependency trước khi chạy

Chạy cell sau trong notebook:

```python
import sys
print("Python:", sys.executable)

from fairness_baselines import (
    check_fair_projection_dependencies,
    check_fair_transport_dependencies,
)

print("FairProjection:", check_fair_projection_dependencies())
print("Fair-transport:", check_fair_transport_dependencies())
```

Kết quả kiểm tra thành công phải hiển thị version của TensorFlow, CVXPY, SciPy,
scikit-learn và danh sách solver. Không tiếp tục full experiment nếu cell này
báo lỗi.

## 7. Thứ tự chạy notebook

1. Restart kernel và chạy cell import.
2. Chạy cell định nghĩa dataset loader.
3. Chạy cell định nghĩa ε-fair algorithm.
4. Chạy các cell định nghĩa FairProjection và Fair-transport experiment.
5. Chạy smoke test trước.
6. Kiểm tra `OptimizerSuccess`, `OptimizerMessage` và các file trong
   `outputs/multiclass_smoke/`.
7. Chỉ sau khi smoke test thành công mới chạy full 30 repetitions.

Smoke test:

```python
smoke_raw, smoke_summary = run_multiclass_experiments(smoke_config)
display(smoke_summary)
```

Full experiment:

```python
multiclass_raw, multiclass_summary = run_multiclass_experiments(full_config)
display(multiclass_summary)
```

Full experiment rất tốn thời gian vì mỗi repetition thực hiện tuning RF/GBM và
chạy nhiều tolerance cho cả FairProjection lẫn Fair-transport.

### Synthetic Figures 1-4 và 10

Các protocol mới nằm trong `reproduction_protocols.py` và được gọi từ notebook:

```python
synthetic_config = SyntheticProtocolConfig(n_repetitions=30)
```

Mỗi Figure có runner riêng. Figure 10 là runner tốn thời gian nhất vì quét:

- `n=100,200,...,2000`, với `N=4000`;
- `N=20,40,...,300`, với `n=2000`;
- `epsilon={0,0.05,0.1,0.15}`;
- 30 repetitions.

Kết quả được lưu trong `outputs/synthetic/`.

### Binary Figures 6-7

```python
binary_config = BinaryProtocolConfig(n_repetitions=30)
```

Mỗi repetition tạo split mới, tune lại RF/GBM và chạy RegLog, RF, GBM, NN,
Fairlearn, epsilon-fair và Fair-adversarial. Grid Fairlearn của paper được truyền
vào `ExponentiatedGradient.eps`:

```text
0.0001, 0.5, 1, 2.5, 5, 10
```

Nó được lưu trong cột `FairlearnTolerance`; cột `Epsilon` chỉ dùng cho phương
pháp epsilon-fair. Kết quả được lưu trong `outputs/binary/`.

Unit test:

```powershell
python -m unittest tests.test_reproduction_protocols -v
```

## 8. Kiểm tra nhanh khi gặp lỗi

### `No module named 'cvxpy'`

Kernel đang chọn sai hoặc `cvxpy` chưa được cài vào chính kernel đó:

```python
import sys
print(sys.executable)
```

Cài vào đúng executable vừa in:

```powershell
"DUONG_DAN_PYTHON_VUA_IN" -m pip install cvxpy
```

### Không import được `FairTransportAdapter`

Restart kernel rồi chạy lại cell import. Notebook cũng đã có `importlib.reload`
để xử lý kernel cache cũ.

### FairProjection thiếu TensorFlow

```powershell
python -m pip install tensorflow
```

### Fair-transport không tìm thấy solver

```powershell
python -m pip install osqp
```

Hoặc cài CBC giống upstream:

```powershell
python -m pip install cylp
```

### TensorFlow in thông báo oneDNN

Đây là thông báo thông tin, không phải lỗi. Sai khác rất nhỏ do thứ tự phép toán
floating-point là bình thường. Seed và version dependency cần được lưu cùng kết
quả thực nghiệm.
