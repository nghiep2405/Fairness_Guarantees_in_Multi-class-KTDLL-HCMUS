
# Mã nguồn tham khảo
- https://github.com/curiousML/epsilon-fairness
- paper: Fairness Guarantees in Multi-class Classification with Demographic Parity - Journal of Machine Learning Research 25 (2024) 1-46

# Kết quả thực nghiệm
- Ứng với thực nghiệm trên synthetic data thì sẽ có folder tên tương ứng trong folder outputs
- Đối với dữ liệu thực tế thì chia ra chạy binary case và multi-class cũng có folder tên tương ứng

# Cách chạy
- Hướng dẫn đầy đủ: [`instruction_run.md`](instruction_run.md)
- py -3.13 -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- pip install numpy pandas scipy matplotlib seaborn scikit-learn lightgbm fairlearn aif360 tensorflow statsmodels cvxpy cylp osqp tqdm
- pip install ipykernel
- python -m ipykernel install --user --name=ktdll-fairness --display-name "Python (KTDLL Fairness)"

## FairProjection
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

## Fair-transport
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

## Protocol thực nghiệm lặp lại

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
