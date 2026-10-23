from pytest import approx

from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_orders_by_summed_reciprocal_rank():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])

    assert fused[0][0] == "b"
    assert fused[0][1] == approx(1 / 62 + 1 / 61)
