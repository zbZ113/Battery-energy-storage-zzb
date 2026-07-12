import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value using the project's canonical form.

    Only JSON-compatible values are accepted. Non-finite floating-point values
    are rejected because they are not part of the JSON data model.
    """

    try:
        return json.dumps(
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


def sha256_canonical(value: Any) -> str:
    """Return SHA-256 over a deterministic UTF-8 JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
