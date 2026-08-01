# Tái lập thực nghiệm Fairness Guarantees in Multi-class Classification

Repository này tái lập các thực nghiệm trong bài báo **Fairness Guarantees in
Multi-class Classification with Demographic Parity**, *Journal of Machine
Learning Research*, 25 (2024), trang 1–46. Trọng tâm của project là khảo sát sự
đánh đổi giữa độ chính xác dự đoán và **Demographic Parity** trong bài toán phân
loại nhị phân và đa lớp, trên cả dữ liệu tổng hợp lẫn dữ liệu thực tế.

Mã nguồn được tổ chức để có thể:

- chạy lại các protocol tương ứng với Figures 1–4, 6–8 và 10;
- so sánh mô hình gốc với các phương pháp hậu xử lý/ràng buộc fairness;
- lặp thí nghiệm với seed cố định và tổng hợp mean/standard deviation;
- lưu dữ liệu thô, bảng tổng hợp, cấu hình, lỗi và biểu đồ để kiểm tra lại kết quả;
- chạy smoke test trước khi thực hiện cấu hình đầy đủ tốn nhiều thời gian.

## Nội dung thực nghiệm

Project gồm ba nhóm thực nghiệm chính:

| Nhóm | Dữ liệu | Nội dung |
|---|---|---|
| Synthetic | Dữ liệu sinh theo phân phối được định nghĩa trong code | Tái lập Figures 1–4 và Figure 10, khảo sát ảnh hưởng của phân phối, cỡ mẫu và mức fairness |
| Binary real-data | DRUG và CRIME | Tái lập Figures 6–7 với các mô hình/baseline phân loại nhị phân |
| Multi-class real-data | DRUG và CRIME sau tiền xử lý, ánh xạ nhãn đa lớp | So sánh mô hình cơ sở, FairProjection và Fair-transport; có chế độ smoke và full |

Metric chính gồm **Accuracy** và empirical **Unfairness**. Unfairness được tính
theo chênh lệch Demographic Parity lớn nhất giữa các nhóm nhạy cảm trên các lớp
dự đoán. Với protocol lặp, project lưu cả kết quả của từng repetition và thống
kê tổng hợp.

## Luồng thực nghiệm

1. Đọc dữ liệu trong `datasets/` hoặc sinh synthetic data.
2. Tiền xử lý đặc trưng, nhãn và thuộc tính nhạy cảm.
3. Chia train/calibration/test theo seed của repetition.
4. Huấn luyện và tuning mô hình cơ sở như Logistic Regression, Random Forest,
   LightGBM hoặc MLP.
5. Áp dụng thuật toán fairness tương ứng: epsilon-fair, Fairlearn,
   AIF360 Adversarial Debiasing, FairProjection hoặc Fair-transport.
6. Đánh giá Accuracy, Unfairness, thời gian chạy và diagnostics của solver.
7. Ghi artifact vào thư mục con tương ứng trong `outputs/`.

## Cấu trúc repository

```text
.
├── experimentation_epsilon_fairness_update.ipynb  # Notebook điều phối thực nghiệm
├── reproduction_protocols.py                       # Protocol lặp cho synthetic/binary
├── configs/
│   └── reproduction.default.txt                    # Cấu hình tham khảo cho full run
├── datasets/
│   ├── drug_consumption.csv                        # Drug Consumption dataset
│   └── communites_and_crime.csv                    # Communities and Crime dataset
├── fairness_baselines/
│   ├── fair_projection_adapter.py                  # Adapter FairProjection
│   ├── fair_transport_adapter.py                   # Adapter Fair-transport
│   ├── aif360_subprocess.py                        # Điều phối AIF360 ở process riêng
│   └── aif360_worker.py                            # Worker TensorFlow v1 cho AIF360
├── third_party/
│   ├── fair_projection/                            # Core FairProjection được vendor
│   └── fair_transport/                             # Core Fair-transport được vendor
├── tests/
│   ├── test_reproduction_protocols.py              # Test protocol và artifact
│   └── test_fair_projection_runtime.py             # Test runtime TF/AIF360 isolation
├── reference/
│   ├── epsilon-fairness_draft_arxiv/               # Mã/notebook tham khảo epsilon-fair
│   └── fair_projection/                            # Script tham khảo FairProjection
└── outputs/
    ├── synthetic/                                  # Figures 1–4 và 10
    ├── binary/                                     # DRUG/CRIME binary, Figures 6–7
    ├── multiclass_smoke/                           # Kết quả kiểm tra nhanh
    └── multiclass/                                 # Kết quả multi-class đầy đủ
```

Các thư mục `reference/` và `third_party/` phục vụ hai mục đích khác nhau:
`reference/` lưu bản tham khảo để đối chiếu, còn `third_party/` chứa phần core
thực sự được adapter của project import khi chạy.

## Thư viện cần thiết

Project được kiểm tra với Python 3.13. Các lệnh cài đặt đầy đủ và version cố định
nằm trong section **Cách chạy** bên dưới. Vai trò của các dependency chính:

| Thư viện | Vai trò |
|---|---|
| `numpy`, `pandas`, `scipy` | Tính toán số, xử lý bảng dữ liệu và tối ưu |
| `scikit-learn` | Tiền xử lý, chia dữ liệu, mô hình cơ sở, tuning và metric |
| `lightgbm` | Mô hình gradient boosting cho các protocol real-data |
| `fairlearn` | `ExponentiatedGradient` với ràng buộc `DemographicParity` |
| `aif360` | Baseline Adversarial Debiasing cho bài toán nhị phân |
| `tensorflow` | Backend FairProjection và worker AIF360 TensorFlow v1 compatibility |
| `cvxpy`, `osqp` | Bài toán tối ưu của FairProjection/Fair-transport và QP solver |
| `statsmodels` | Các tiện ích thống kê được dùng trong môi trường thực nghiệm |
| `matplotlib`, `seaborn` | Sinh và trình bày biểu đồ |
| `tqdm` | Theo dõi tiến độ các vòng lặp tối ưu |
| `ipykernel`, `jupyter-client`, `notebook` | Chạy notebook bằng kernel riêng của project |
| `cylp` (tùy chọn) | Bổ sung CBC solver, được Fair-transport ưu tiên nếu có |

Không cần tải dataset sau khi clone vì hai file dữ liệu đầu vào đã có trong
`datasets/`. Không cần cài package riêng cho code trong `fairness_baselines/`,
`reproduction_protocols.py` hoặc `third_party/`; các module này được import trực
tiếp khi chạy từ thư mục gốc repository.

# Cách chạy

Toàn bộ lệnh dưới đây cần được chạy tại thư mục gốc của repository. Khuyến nghị
dùng Python 3.13, là phiên bản đã được kiểm tra với mã nguồn hiện tại.

## 1. Cài đặt thư viện
### 1.1 Cài đặt trên Windows

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

### 1.2 Cài đặt trên Linux

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

## 2. Kiểm tra môi trường

Kiểm tra các package chính và solver của CVXPY:

```bash
python -c "import numpy, pandas, scipy, sklearn, lightgbm, fairlearn, tensorflow, cvxpy; print('Dependencies: OK'); print('CVXPY solvers:', cvxpy.installed_solvers())"
```

Chạy unit test trước khi thực hiện các thí nghiệm:

```bash
python -m unittest tests.test_reproduction_protocols tests.test_fair_projection_runtime -v
```

Các test phải kết thúc với trạng thái `OK`.

## 3. Mở và chạy notebook

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

## 4. Kết quả đầu ra

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

## 5. Solver tùy chọn cho Fair-transport

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

## 6. Xử lý lỗi thường gặp

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

# Các phương pháp fairness được tích hợp

### Epsilon-fairness

Đây là phương pháp trung tâm của bài báo được tái lập. Phần triển khai và luồng
thực nghiệm chính nằm trong notebook, kết hợp các hàm protocol lặp trong
`reproduction_protocols.py`. Các mức epsilon được quét để quan sát đường đánh
đổi giữa Accuracy và Demographic Parity.

### Fairlearn

Baseline Fairlearn sử dụng `ExponentiatedGradient` và constraint
`DemographicParity`. Grid theo protocol là
`{0.0001, 0.5, 1, 2.5, 5, 10}` cho tham số `ExponentiatedGradient.eps`.
Giá trị này được ghi trong artifact dưới tên `FairlearnTolerance`; nó không được
dùng như `DemographicParity.difference_bound`.

### AIF360 Adversarial Debiasing

AIF360 Adversarial Debiasing cần TensorFlow v1 graph mode. Graph mode là trạng
thái process-wide và có thể làm FairProjection không còn chạy được bằng eager
mode. Vì vậy baseline này luôn được gọi qua
`fairness_baselines/aif360_subprocess.py`, sau đó thực thi trong worker riêng
`fairness_baselines/aif360_worker.py`. Cách ly process giúp có thể chạy tiếp
FairProjection trong cùng phiên notebook.

### FairProjection

Core được vendor tại `third_party/fair_projection/`; adapter an toàn nằm tại
`fairness_baselines/fair_projection_adapter.py`. Adapter tái sử dụng mô hình cơ
sở đã fit và cùng tập split/probability với các phương pháp khác để bảo đảm phép
so sánh nhất quán.

Cấu hình full multi-class mặc định:

- constraint: Statistical Parity (`sp`);
- divergence: cross-entropy;
- `alpha={0, 0.1, 0.2, 0.5, 0.75}`;
- `rho=2`, `max_iter=500`;
- backend: `method="tf"`.

Backend TensorFlow yêu cầu eager mode. Adapter kiểm tra điều kiện này trước khi
chạy ADMM. Có thể đặt `fair_projection_method="np"` để dùng fallback
NumPy/CVXPY; nhánh này không import TensorFlow. Smoke test chỉ chạy một ADMM
iteration để xác nhận dependency và wiring, không dùng kết quả đó trong báo cáo.

### Fair-transport

Fair-transport là phương pháp hậu xử lý Wasserstein-barycenter từ bài báo *Fair
and Optimal Classification via Post-Processing* (Xian, Yin và Zhao, ICML 2023).
Core nằm tại `third_party/fair_transport/postprocess.py`; adapter nằm tại
`fairness_baselines/fair_transport_adapter.py`.

Full multi-class run sử dụng cùng base probabilities và calibration/test split
với các baseline khác, với `alpha={0, 0.1, 0.2, 0.3, 0.4}`. Chế độ solver
`AUTO` ưu tiên CBC cho LP và OSQP cho QP. Nếu CBC chưa được cài, adapter chọn LP
solver tương thích khác từ CVXPY và ghi solver thực tế vào diagnostics.

# Protocol và cấu hình tái lập

`reproduction_protocols.py` cung cấp các runner độc lập cho:

- synthetic Figure 1;
- synthetic Figures 2–3 và phần tổng hợp phân phối của Figure 3;
- synthetic Figure 4;
- synthetic Figure 10;
- DRUG/CRIME binary Figures 6–7.

Cấu hình mặc định chạy 30 repetitions với master seed cố định. Mỗi repetition
sinh seed riêng, huấn luyện lại mô hình và ghi thời gian chạy. File
`configs/reproduction.default.txt` mô tả cấu hình tham khảo cho nhánh
FairProjection + LightGBM trên DRUG/CRIME, bao gồm:

- số lớp dự kiến của từng dataset;
- không gian `RandomizedSearchCV` cho LightGBM;
- tham số FairProjection;
- pilot seeds `0–4` và full seeds `0–29`;
- tỷ lệ train/calibration/test đề xuất `0.6/0.2/0.2`.

Các dòng đánh dấu `[CHỜ CHỐT — nhóm]` trong file cấu hình là quyết định thực
nghiệm chưa được paper hoặc upstream quy định trực tiếp. Cần chốt mapping nhãn,
thuộc tính nhạy cảm, scorer và split trước khi tạo kết quả cuối cùng để báo cáo.

# Dữ liệu

| File cục bộ | Nguồn | Mục đích |
|---|---|---|
| `datasets/drug_consumption.csv` | [Drug Consumption Classification trên Kaggle](https://www.kaggle.com/datasets/mexwell/drug-consumption-classification) | Thực nghiệm DRUG binary và multi-class |
| `datasets/communites_and_crime.csv` | [Communities and Crime trên Kaggle](https://www.kaggle.com/datasets/chaditya95/communities-and-crime-data-set/data) | Thực nghiệm CRIME binary và multi-class |

Tên file `communites_and_crime.csv` được giữ theo repository hiện tại (từ
`communites` thiếu chữ `i`). Không đổi tên file nếu chưa cập nhật đồng thời các
cell đọc dữ liệu trong notebook.

# Artifact kết quả

Tùy protocol, `outputs/` có thể chứa các loại file sau:

| Hậu tố/tên file | Ý nghĩa |
|---|---|
| `*_raw.csv` | Kết quả từng repetition, từng method và mức tolerance/alpha |
| `*_summary.csv` | Mean, standard deviation và số repetition theo nhóm |
| `*_config.json` | Cấu hình đã dùng để tạo kết quả |
| `*_failures.csv` | Lỗi của từng cấu hình trong multi-class run |
| file hình | Biểu đồ tái lập figure tương ứng nếu runner sinh biểu đồ |

Không nên chỉ dựa vào summary khi một run có lỗi. Hãy kiểm tra thêm
`multiclass_failures.csv`, số repetition trong summary và config JSON để xác
nhận kết quả đầy đủ.

# Kiểm thử

Hai test module có mục tiêu khác nhau:

- `tests.test_reproduction_protocols` kiểm tra metric, schema kết quả, khả năng
  tái lập và quy trình lưu artifact;
- `tests.test_fair_projection_runtime` kiểm tra FairProjection, dependency
  preflight và việc AIF360 graph mode được cách ly khỏi process chính.

Test là bước kiểm tra nhanh về code path, không thay thế full experiment 30
repetitions. Trước khi chạy full, nên chạy cả unit test và multi-class smoke test.

# Nguồn tham khảo

- [Repository epsilon-fairness](https://github.com/curiousML/epsilon-fairness)
- *Fairness Guarantees in Multi-class Classification with Demographic Parity*,
  Journal of Machine Learning Research 25 (2024), 1–46.
- [Repository FairProjection của nhóm tái lập](https://github.com/khanhchung101/ktdll-fairprojection)
- [Repository upstream fair-classification](https://github.com/uiuctml/fair-classification)
- Ruicheng Xian, Lang Yin và Han Zhao, *Fair and Optimal Classification via
  Post-Processing*, ICML 2023.

Mã Fair-transport được vendor từ tag `icml.23`, commit
`ff83c13c3c17de95ac7a29c0889727665014a08a`. Các thay đổi cục bộ tập trung vào
validation, lựa chọn/fallback solver và diagnostics; thông tin chi tiết nằm
trong `third_party/fair_transport/README.md` và các file `LICENSE` tương ứng.
