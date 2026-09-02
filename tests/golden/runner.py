from tests.golden.cases import CASES


def main() -> int:
    code = 0
    for name, fn in CASES:
        ok = False
        try:
            ok = bool(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"{name} FAIL ({exc})")
            code = 1
            continue
        print(f"{name} {'PASS' if ok else 'FAIL'}")
        if not ok:
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
