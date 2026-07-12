import hashlib
import json
from typing import Any


def sha256_canonical(value: Any) -> str:
    """Return SHA-256 over a deterministic UTF-8 JSON representation.

    Only JSON-compatible values are accepted. Non-finite floating-point values
    are rejected because they are not part of the JSON data model.
    """

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except ValueError as exc:
        raise ValueError("Canonical JSON requires finite numeric values") from exc
    except (TypeError, OverflowError) as exc:
        raise TypeError("Value must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()
