"""Minimal smoke test — package imports cleanly."""


def test_import_package() -> None:
    import app

    assert app.__version__ == "0.1.0"
