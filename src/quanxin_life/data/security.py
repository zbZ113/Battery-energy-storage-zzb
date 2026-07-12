from pathlib import Path

UNSAFE_EXTERNAL_SUFFIXES = frozenset({".pkl", ".pickle", ".joblib", ".pth", ".pt"})


def assert_safe_external_data_file(path: Path) -> None:
    if path.suffix.lower() in UNSAFE_EXTERNAL_SUFFIXES:
        raise ValueError(f"unsafe serialized artifact is not accepted: {path.name}")

