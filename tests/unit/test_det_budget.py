"""Unit tests for detection.budget (K <-> bm### tag)."""
import pytest

from detection.budget import budget_tag, parse_budget_tag, is_budget_tag


@pytest.mark.parametrize("K,tag", [(0.05, "bm005"), (0.25, "bm025"), (0.5, "bm050"),
                                   (1.0, "bm100"), (2.0, "bm200")])
def test_roundtrip(K, tag):
    assert budget_tag(K) == tag
    assert parse_budget_tag(tag) == pytest.approx(K)


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        budget_tag(0.0)
    with pytest.raises(ValueError):
        budget_tag(10.0)


def test_parse_invalid_raises():
    for bad in ["bm05", "bm0500", "xx050", "", "bm05a"]:
        with pytest.raises(ValueError):
            parse_budget_tag(bad)


def test_is_budget_tag():
    assert is_budget_tag("bm050")
    assert not is_budget_tag("bm5")
    assert not is_budget_tag("seed42")
