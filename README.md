
# Mã nguồn tham khảo
- https://github.com/curiousML/epsilon-fairness
- paper: Fairness Guarantees in Multi-class Classification with Demographic Parity - Journal of Machine Learning Research 25 (2024) 1-46

# Cách chạy
- python -m venv venv
- .\venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- pip install numpy pandas scipy matplotlib seaborn scikit-learn lightgbm fairlearn aif360 tensorflow statsmodels
- pip install ipykernel
- python -m ipykernel install --user --name=venv --display-name "Python (epsilon-fairness)" 

# Dataset
- https://www.kaggle.com/datasets/mexwell/drug-consumption-classification
- https://www.kaggle.com/datasets/chaditya95/communities-and-crime-data-set/data