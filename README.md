<<<<<<< HEAD
# Movie Recommender Research

Repo này hiện được chia thành hai khối rõ ràng:

- `AI/`: khu vực nghiên cứu, huấn luyện, đánh giá và lưu artifact mô hình
- phần top-level còn lại: khu vực hệ thống suy luận, API và UI demo

Mục tiêu hiện tại:
- hiểu rõ `MovieLens 1M`
- huấn luyện và đánh giá `Matrix Factorization`
- huấn luyện và đánh giá `NCF/NeuMF`
- so sánh công bằng giữa học máy truyền thống và học sâu

Giai đoạn sau:
- chọn mô hình phù hợp nhất
- đóng gói pipeline suy luận
- xây API và giao diện người dùng

Hiện tại repo đã bắt đầu giai đoạn triển khai:
- dùng `Matrix Factorization` làm mô hình suy luận chính
- có service recommend top-N từ artifact đã train
- có `FastAPI` để phục vụ gợi ý cho người dùng

## Cấu trúc chính

- `data/`: dữ liệu dùng chung cho cả training và serving
- `AI/`: toàn bộ phần nghiên cứu mô hình
- `AI/notebooks/`: EDA, train MF, train NCF/NeuMF, so sánh mô hình
- `AI/src/`: mã nguồn cho train, eval, metric và model research
- `AI/artifacts/`: trọng số mô hình, metric, hình ảnh kết quả
- `AI/reports/`: ghi chú nghiên cứu và báo cáo LaTeX
- `src/`: mã nguồn runtime cho phần serving và inference
- `api/`: FastAPI app cho giai đoạn hệ thống
- `ui/`: giao diện Streamlit demo
- `scripts/`: script chạy thử nhanh ngoài notebook

Bạn có thể xem mô tả chi tiết riêng của phần research tại [AI/README.md](/Users/phamvantu/Desktop/TTCS-Refactor/movie-recomender-search/AI/README.md).

## Thứ tự làm việc đề xuất

1. Tải `MovieLens 1M` chính thức vào `data/raw/ml-1m/`
2. Chạy `AI/notebooks/01_eda_movielens_1m.ipynb`
3. Chạy `AI/notebooks/02_matrix_factorization.ipynb`
4. Chạy `AI/notebooks/03_ncf_neumf.ipynb`
5. Chạy `AI/notebooks/04_model_comparison.ipynb`

## Dataset

Đặt các file chính thức của MovieLens 1M vào:

```text
data/raw/ml-1m/
├── users.dat
├── movies.dat
└── ratings.dat
```

Nguồn chính thức:
- https://grouplens.org/datasets/movielens/1m/
- https://files.grouplens.org/datasets/movielens/ml-1m.zip

## Gợi ý môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name movie-recommender-research
```

## Workflow Colab

Nếu bạn muốn:
- copy repo ra thư mục local của Colab như `/content/movie-recomender-search`
- nhưng vẫn đọc dữ liệu và lưu toàn bộ output về Google Drive

thì có thể dùng các biến môi trường sau trước khi chạy notebook:

```python
import os

os.environ["MOVIE_REC_RAW_DIR"] = "/content/drive/MyDrive/movie-recomender-search/data/raw/ml-1m"
os.environ["MOVIE_REC_ARTIFACTS_DIR"] = "/content/drive/MyDrive/movie-recomender-search/AI/artifacts"
os.environ["MOVIE_REC_MODELS_DIR"] = "/content/drive/MyDrive/movie-recomender-search/AI/artifacts/models"
os.environ["MOVIE_REC_METRICS_DIR"] = "/content/drive/MyDrive/movie-recomender-search/AI/artifacts/metrics"
os.environ["MOVIE_REC_FIGURES_DIR"] = "/content/drive/MyDrive/movie-recomender-search/AI/artifacts/figures"
```

Khi đó:
- code vẫn chạy từ bản copy trong `/content`
- dữ liệu vẫn đọc từ Drive
- metric, model và hình ảnh vẫn được lưu về Drive

## Chạy thử suy luận local

Sau khi đã có `AI/artifacts/models/mf_model.npz`, bạn có thể kiểm tra nhanh recommendation:

```bash
python scripts/demo_recommend.py 1 --top-k 5
```

## Chạy API local

```bash
uvicorn api.main:app --reload
```

Các endpoint chính:
- `GET /health`
- `GET /model-info`
- `GET /popular?top_k=10`
- `GET /recommend/{user_id}?top_k=10`
- `GET /recommend/{user_id}/explain/{movie_id}`

## Chạy UI demo với Streamlit

Mở terminal thứ nhất để chạy API:

```bash
uvicorn api.main:app --reload
```

Mở terminal thứ hai để chạy giao diện:

```bash
streamlit run ui/streamlit_app.py
```

Nếu API chạy ở host hoặc port khác:

```bash
MOVIE_REC_API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py
```

UI hiện hỗ trợ:
- lấy recommendation cá nhân hóa theo `user_id`
- xem danh sách phim phổ biến
- xem metadata mô hình đang phục vụ
- giải thích ngắn gọn vì sao một phim được gợi ý
=======
# movie-recomender-search
>>>>>>> cf89782493fc54a386c94e7e08ff938e64eba022
