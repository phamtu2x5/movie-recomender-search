# AI Workspace

Thư mục này gom toàn bộ phần nghiên cứu mô hình của dự án vào một chỗ riêng, tách khỏi phần hệ thống chạy thực tế.

## Bên trong có gì

- `notebooks/`: notebook EDA, train `Matrix Factorization`, train `NeuMF`, và so sánh kết quả
- `src/`: mã nguồn phục vụ train, evaluate, metric, feature encoding và model research
- `artifacts/`: model đã train, metric JSON/CSV, và figure sinh ra từ thí nghiệm
- `reports/`: ghi chú và tài liệu LaTeX cho báo cáo
- `tests/`: test cho phần pipeline nghiên cứu

## Luồng làm việc

1. Đặt `MovieLens 1M` vào `../data/raw/ml-1m/`
2. Chạy `notebooks/01_eda_movielens_1m.ipynb`
3. Chạy `notebooks/02_matrix_factorization.ipynb`
4. Chạy `notebooks/03_ncf_neumf.ipynb`
5. Chạy `notebooks/04_model_comparison.ipynb`
6. Dùng artifact tốt nhất trong `artifacts/models/` cho phần hệ thống

## Quan hệ với phần system

- Phần `AI/` chịu trách nhiệm huấn luyện và đánh giá mô hình.
- Phần top-level `src/`, `api/`, `ui/`, `scripts/` chịu trách nhiệm suy luận, phục vụ API, và demo hệ thống.
- Hiện tại hệ thống đang dùng model `Matrix Factorization` đã lưu tại `AI/artifacts/models/mf_model.npz`.
