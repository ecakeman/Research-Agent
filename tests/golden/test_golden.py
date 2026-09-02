from tests.golden.cases import CASES


def test_golden_cases():
    failed = []
    for name, fn in CASES:
        if not fn():
            failed.append(name)
    assert failed == []
