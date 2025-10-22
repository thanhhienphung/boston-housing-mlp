# Hướng dẫn nhanh cho người mới

- Mục tiêu: chạy script tuning MLP trên Boston Housing dataset.

Files liên quan:
- [requirements.txt](requirements.txt)
- [resource/BostonHousing.csv](resource/BostonHousing.csv)
- [src/mlp_tuning/mlp_tuning_boston.py](src/mlp_tuning/mlp_tuning_boston.py) (hàm [`mlp_tuning.mlp_tuning_boston.bostonHousingData`](src/mlp_tuning/mlp_tuning_boston.py), [`mlp_tuning.mlp_tuning_boston.mlp_tuning_boston_MLPRegressor`](src/mlp_tuning/mlp_tuning_boston.py), [`mlp_tuning.mlp_tuning_boston.mlp_tuning_boston_MLPClassifier`](src/mlp_tuning/mlp_tuning_boston.py))

1) Cài đặt phụ thuộc (từ project root)
```sh
python -m venv .venv         # (tuỳ chọn) tạo virtualenv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r [requirements.txt](http://_vscodecontentref_/0)

2) Xem dataset
File: resource/BostonHousing.csv

3) Chạy code
cd src/mlp_tuning
python -m mlp_tuning_boston.py