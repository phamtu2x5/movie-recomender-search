import numpy as np

from src.inference.mf_loader import MFArtifactBundle


def test_predict_for_user_returns_score_per_known_movie() -> None:
    bundle = MFArtifactBundle(
        user_factors=np.array([[1.0, 0.5]]),
        item_factors=np.array([[0.2, 0.1], [0.6, 0.4]]),
        user_bias=np.array([0.1]),
        item_bias=np.array([0.05, 0.02]),
        global_bias=3.0,
        user_ids=np.array([10]),
        item_ids=np.array([100, 200]),
    )

    preds = bundle.predict_for_user(10, [100, 200])

    assert preds.shape == (2,)
    assert preds[1] > preds[0]


def test_predict_single_and_similarity_support_known_items() -> None:
    bundle = MFArtifactBundle(
        user_factors=np.array([[1.0, 0.0]]),
        item_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        user_bias=np.array([0.0]),
        item_bias=np.array([0.0, 0.0]),
        global_bias=3.0,
        user_ids=np.array([1]),
        item_ids=np.array([10, 20]),
    )

    assert bundle.has_item(10) is True
    assert bundle.predict_single(1, 10) == 4.0
    assert bundle.item_similarity(10, 20) == 0.0
