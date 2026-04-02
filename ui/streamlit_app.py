import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.environ.get("MOVIE_REC_API_URL", "http://127.0.0.1:8000").rstrip("/")


st.set_page_config(
    page_title="Movie Recommender Demo",
    page_icon="🎬",
    layout="wide",
)


if "recommend_payload" not in st.session_state:
    st.session_state["recommend_payload"] = None
if "recommend_error" not in st.session_state:
    st.session_state["recommend_error"] = None
if "explain_payload" not in st.session_state:
    st.session_state["explain_payload"] = None
if "explain_error" not in st.session_state:
    st.session_state["explain_error"] = None
if "popular_payload" not in st.session_state:
    st.session_state["popular_payload"] = None
if "popular_error" not in st.session_state:
    st.session_state["popular_error"] = None
if "model_info_payload" not in st.session_state:
    st.session_state["model_info_payload"] = None
if "model_info_error" not in st.session_state:
    st.session_state["model_info_error"] = None


def fetch_health() -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return True, response.json(), None
    except requests.RequestException as exc:
        return False, None, str(exc)


def fetch_recommendations(user_id: int, top_k: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/recommend/{user_id}",
            params={"top_k": top_k},
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def fetch_popular(top_k: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/popular",
            params={"top_k": top_k},
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def fetch_model_info() -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def fetch_explanation(user_id: int, movie_id: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/recommend/{user_id}/explain/{movie_id}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def render_recommendation_cards(recommendations: list[dict[str, Any]]) -> None:
    if not recommendations:
        st.info("Không có gợi ý nào được trả về.")
        return

    for rec in recommendations:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**#{rec['rank']} - {rec['title']}**")
                st.caption(rec["genres"])
            with right:
                st.metric("Score", f"{rec['score']:.4f}")


st.title("Movie Recommender Demo")
st.caption("Demo hệ thống gợi ý phim sử dụng Matrix Factorization đã huấn luyện trên MovieLens 1M")

with st.sidebar:
    st.subheader("API")
    st.code(API_BASE_URL)
    ok, health_payload, health_error = fetch_health()
    if ok and health_payload:
        st.success("API đang hoạt động")
        st.write(
            {
                "users": health_payload["n_users"],
                "movies": health_payload["n_movies"],
                "model_users": health_payload["n_known_model_users"],
                "model_items": health_payload["n_known_model_items"],
            }
        )
    else:
        st.error("Không kết nối được API")
        if health_error:
            st.caption(health_error)

st.markdown(
    """
    Nhập `user_id` để lấy gợi ý cá nhân hóa.
    Nếu `user_id` không nằm trong mô hình đã huấn luyện, hệ thống sẽ tự động fallback sang danh sách phim phổ biến.
    """
)

col1, col2 = st.columns([2, 1])
with col1:
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
with col2:
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10, step=1)

recommend_tab, popular_tab, info_tab = st.tabs(["Personalized", "Popular", "Model Info"])

with recommend_tab:
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        payload, error = fetch_recommendations(int(user_id), int(top_k))
        st.session_state["recommend_payload"] = payload
        st.session_state["recommend_error"] = error
        st.session_state["explain_payload"] = None
        st.session_state["explain_error"] = None

    payload = st.session_state["recommend_payload"]
    error = st.session_state["recommend_error"]

    if error:
        st.error(f"Lỗi gọi API: {error}")
    elif payload:
        strategy = payload["strategy"]
        known_user = payload["known_user"]
        if known_user:
            st.success(f"Đã dùng chiến lược: `{strategy}`")
        else:
            st.warning(f"User chưa có trong mô hình, dùng fallback: `{strategy}`")

        recommendations = payload.get("recommendations", [])
        st.subheader("Top Recommendations")
        render_recommendation_cards(recommendations)

        if recommendations:
            explain_options = {
                f"#{rec['rank']} - {rec['title']}": rec["movie_id"]
                for rec in recommendations
            }
            selected_label = st.selectbox(
                "Chọn một phim để xem giải thích",
                options=list(explain_options.keys()),
            )
            if st.button("Explain Recommendation", use_container_width=True):
                explain_payload, explain_error = fetch_explanation(
                    user_id=int(payload["user_id"]),
                    movie_id=int(explain_options[selected_label]),
                )
                st.session_state["explain_payload"] = explain_payload
                st.session_state["explain_error"] = explain_error

        explain_payload = st.session_state["explain_payload"]
        explain_error = st.session_state["explain_error"]

        if explain_error:
            st.error(f"Lỗi gọi explain API: {explain_error}")
        elif explain_payload:
            st.subheader("Explanation")
            st.write(explain_payload["reason"])
            meta_cols = st.columns(3)
            meta_cols[0].metric("Known User", "Yes" if explain_payload["known_user"] else "No")
            meta_cols[1].metric(
                "Predicted Score",
                "-" if explain_payload["predicted_score"] is None else f"{explain_payload['predicted_score']:.4f}",
            )
            meta_cols[2].metric(
                "Popular Rank",
                "-" if explain_payload["popular_rank"] is None else str(explain_payload["popular_rank"]),
            )

            if explain_payload["genre_overlap"]:
                st.caption(
                    "Genre overlap với lịch sử user: "
                    + ", ".join(explain_payload["genre_overlap"])
                )

            supporting_movies = explain_payload.get("supporting_movies", [])
            if supporting_movies:
                st.markdown("**Supporting movies from user history**")
                for movie in supporting_movies:
                    st.write(
                        f"- {movie['title']} ({movie['genres']}) | similarity={movie['similarity']:.4f}"
                    )

            with st.expander("Raw explanation payload"):
                st.json(explain_payload)

        with st.expander("Raw recommendation payload"):
            st.json(payload)

with popular_tab:
    if st.button("Get Popular Movies", use_container_width=True):
        payload, error = fetch_popular(int(top_k))
        st.session_state["popular_payload"] = payload
        st.session_state["popular_error"] = error

    payload = st.session_state["popular_payload"]
    error = st.session_state["popular_error"]
    if error:
        st.error(f"Lỗi gọi API: {error}")
    elif payload:
        st.success("Đã tải danh sách phim phổ biến")
        render_recommendation_cards(payload.get("recommendations", []))
        with st.expander("Raw popular payload"):
            st.json(payload)

with info_tab:
    if st.button("Load Model Info", use_container_width=True):
        payload, error = fetch_model_info()
        st.session_state["model_info_payload"] = payload
        st.session_state["model_info_error"] = error

    payload = st.session_state["model_info_payload"]
    error = st.session_state["model_info_error"]
    if error:
        st.error(f"Lỗi gọi API: {error}")
    elif payload:
        st.json(payload)
