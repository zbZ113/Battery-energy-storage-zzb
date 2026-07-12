import hashlib
import stat
import struct
import warnings
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pytest

import quanxin_life.data.hust_archive as hust_archive
from quanxin_life.data.hust_archive import (
    ExpectedHustMember,
    HustArchiveLimits,
    audit_hust_archive,
)
from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalogEntry


def _source(**updates: object) -> SourceCatalogEntry:
    values: dict[str, object] = {
        "dataset_id": "HUST",
        "version": "Mendeley-v2",
        "source_uri": "https://data.mendeley.com/datasets/nsc7hnsg4s/2",
        "paper_uri": "https://doi.org/10.1039/D2EE01676A",
        "license_status": "CC BY 4.0",
        "ingestion_mode": IngestionMode.QUARANTINE_CONVERSION,
        "expected_suffixes": (".zip",),
        "prohibited_direct_suffixes": (".pkl", ".pickle"),
    }
    values.update(updates)
    return SourceCatalogEntry.model_validate(values)


def _write_archive(
    path: Path,
    members: Mapping[str, bytes],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    def raw_name_info(name: str) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name)
        # ZipInfo normalizes os.sep on construction. Restoring these attributes
        # lets the test create the exact untrusted central-directory name.
        info.filename = name
        info.orig_filename = name
        info.compress_type = compression
        return info

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(raw_name_info("our_data/"), b"")
        for name, payload in members.items():
            archive.writestr(raw_name_info(name), payload)


def _manifest(path: Path, **updates: object) -> RawFileManifest:
    values: dict[str, object] = {
        "dataset_id": "HUST",
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_uri": "https://data.mendeley.com/datasets/nsc7hnsg4s/2",
        "license_name": "CC BY 4.0",
    }
    values.update(updates)
    return RawFileManifest.model_validate(values)


def _mark_first_file_encrypted(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    local = payload.find(b"PK\x03\x04")
    central = payload.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    local_flags = struct.unpack_from("<H", payload, local + 6)[0]
    central_flags = struct.unpack_from("<H", payload, central + 8)[0]
    struct.pack_into("<H", payload, local + 6, local_flags | 0x1)
    struct.pack_into("<H", payload, central + 8, central_flags | 0x1)
    path.write_bytes(payload)


def test_inventory_mode_streams_members_without_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "our_data.zip"
    payload = b"opaque pickle bytes; never deserialize"
    _write_archive(path, {"our_data/1-1.pkl": payload})

    def extraction_is_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("archive extraction must not be called")

    monkeypatch.setattr(zipfile.ZipFile, "extract", extraction_is_forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extraction_is_forbidden)

    audit = audit_hust_archive(path, _manifest(path), _source())

    assert audit.dataset_id == "HUST"
    assert audit.version == "Mendeley-v2"
    assert audit.paper_uri == _source().paper_uri
    assert audit.raw_relative_path == _manifest(path).relative_path
    assert audit.archive_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit.member_count == 1
    assert audit.total_uncompressed_size == len(payload)
    assert audit.members[0].path == "our_data/1-1.pkl"
    assert audit.members[0].sha256 == hashlib.sha256(payload).hexdigest()
    assert audit.ready_for_conversion is False
    assert audit.inventory_version is None
    assert audit.warnings
    assert audit.audited_at.utcoffset() is not None


def test_enforced_inventory_must_match_exactly(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    payloads = {"our_data/1-1.pkl": b"cell one", "our_data/2-1.pkl": b"cell two"}
    _write_archive(path, payloads)
    expected = tuple(
        ExpectedHustMember(path=name, sha256=hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    )

    audit = audit_hust_archive(
        path,
        _manifest(path),
        _source(),
        expected_inventory=expected,
        inventory_version="hust-mendeley-v2-sha256-v1",
    )

    assert audit.ready_for_conversion is True
    assert audit.inventory_version == "hust-mendeley-v2-sha256-v1"
    assert audit.warnings == ()


def test_rejects_outer_archive_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_hust_archive(path, _manifest(path, sha256="0" * 64), _source())


def test_rejects_outer_size_before_verifying_raw_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})
    verify_called = False

    def fail_if_called(*args: object, **kwargs: object) -> str:
        nonlocal verify_called
        verify_called = True
        raise AssertionError("verify_raw_file_stream must not run before the outer size check")

    monkeypatch.setattr(hust_archive, "verify_raw_file_stream", fail_if_called, raising=False)

    with pytest.raises(ValueError, match="outer size limit"):
        hust_archive.audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_archive_size=1),
        )

    assert verify_called is False


def test_uses_one_open_handle_for_manifest_verification_and_zip_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})
    manifest = _manifest(path)
    verifier_handle: object | None = None
    zip_handle: object | None = None
    original_zip_init = zipfile.ZipFile.__init__

    def verify_from_open_handle(
        handle: object, archive_path: Path, raw_manifest: RawFileManifest
    ) -> str:
        nonlocal verifier_handle
        assert archive_path == path
        assert raw_manifest == manifest
        verifier_handle = handle
        return raw_manifest.sha256

    def record_zip_handle(
        archive: zipfile.ZipFile, file: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal zip_handle
        zip_handle = file
        original_zip_init(archive, file, *args, **kwargs)

    monkeypatch.setattr(
        hust_archive, "verify_raw_file_stream", verify_from_open_handle, raising=False
    )
    monkeypatch.setattr(zipfile.ZipFile, "__init__", record_zip_handle)

    audit_hust_archive(path, manifest, _source())

    assert verifier_handle is not None
    assert zip_handle is verifier_handle


def test_closed_stream_verifier_handle_blocks_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})
    manifest = _manifest(path)

    def close_verified_handle(
        handle: object, archive_path: Path, raw_manifest: RawFileManifest
    ) -> str:
        assert archive_path == path
        assert raw_manifest == manifest
        handle.close()  # type: ignore[attr-defined]
        return raw_manifest.sha256

    monkeypatch.setattr(
        hust_archive, "verify_raw_file_stream", close_verified_handle, raising=False
    )

    with pytest.raises(ValueError, match="closed file"):
        audit_hust_archive(path, manifest, _source())


def test_uses_fstat_for_the_outer_archive_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})
    manifest = _manifest(path)
    original_stat = Path.stat

    def path_stat_is_forbidden(self: Path, *args: object, **kwargs: object) -> object:
        if self == path:
            raise AssertionError("the archive path must not be stat'ed after opening")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", path_stat_is_forbidden)

    assert audit_hust_archive(path, manifest, _source()).member_count == 1


@pytest.mark.parametrize("payload", [b"not a zip", b"PK\x03\x04truncated"])
def test_rejects_non_zip_or_broken_central_directory(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "our_data.zip"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="valid ZIP central directory"):
        audit_hust_archive(path, _manifest(path), _source())


def test_rejects_archive_member_count_and_total_size_limits(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"1234", "our_data/2-1.pkl": b"5678"})

    with pytest.raises(ValueError, match="outer size limit"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_archive_size=1),
        )

    with pytest.raises(ValueError, match="member count limit"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_members=2),
        )

    with pytest.raises(ValueError, match="total uncompressed size"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_total_uncompressed_size=7),
        )


def test_rejects_encrypted_member_flag(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"}, compression=zipfile.ZIP_STORED)
    _mark_first_file_encrypted(path)

    with pytest.raises(ValueError, match="encrypted ZIP member"):
        audit_hust_archive(path, _manifest(path), _source())


@pytest.mark.parametrize(
    "source_updates, manifest_updates, match",
    [
        ({"dataset_id": "MATR"}, {}, "HUST"),
        ({"ingestion_mode": IngestionMode.TABULAR}, {}, "quarantine_conversion"),
        ({"source_uri": "https://example.invalid"}, {}, "source URI"),
        ({}, {"license_name": "unknown"}, "license"),
    ],
)
def test_rejects_wrong_source_binding(
    tmp_path: Path,
    source_updates: dict[str, object],
    manifest_updates: dict[str, object],
    match: str,
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    with pytest.raises(ValueError, match=match):
        audit_hust_archive(
            path,
            _manifest(path, **manifest_updates),
            _source(**source_updates),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "../our_data.zip",
        "archives/our_data.zip",
        "/raw/our_data.zip",
        r"C:\\raw\\our_data.zip",
    ],
)
def test_rejects_unsafe_manifest_relative_path(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    with pytest.raises(ValueError, match="safe simple basename"):
        audit_hust_archive(path, _manifest(path, relative_path=relative_path), _source())


def test_rejects_manifest_path_with_a_different_archive_name(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    with pytest.raises(ValueError, match="does not match archive path name"):
        audit_hust_archive(path, _manifest(path, relative_path="other_data.zip"), _source())


@pytest.mark.parametrize("prohibited_direct_suffixes", [(".pkl",), (".pickle",)])
def test_requires_both_pickle_suffixes_in_source_catalog(
    tmp_path: Path, prohibited_direct_suffixes: tuple[str, ...]
) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    with pytest.raises(ValueError, match=r"prohibit direct \.pkl and \.pickle ingestion"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(prohibited_direct_suffixes=prohibited_direct_suffixes),
        )


def test_accepts_case_insensitive_pickle_suffix_prohibitions(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})

    audit = audit_hust_archive(
        path,
        _manifest(path),
        _source(prohibited_direct_suffixes=(".PKL", ".PickLe")),
    )

    assert audit.member_count == 1


@pytest.mark.parametrize(
    "member",
    [
        "../escape.pkl",
        "/our_data/escape.pkl",
        "C:/our_data/escape.pkl",
        "our_data\\escape.pkl",
        "other/escape.pkl",
        "our_data//escape.pkl",
        "our_data/./escape.pkl",
        "our_data/a/escape.pkl",
    ],
)
def test_rejects_unsafe_or_out_of_root_member_paths(tmp_path: Path, member: str) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {member: b"cell"})

    with pytest.raises(ValueError, match="member path"):
        audit_hust_archive(path, _manifest(path), _source())


@pytest.mark.parametrize(
    "member",
    [
        "our_data/COM1 .pkl",
        "our_data/COM1..pkl",
        "our_data/AUX .pkl",
        "our_data/COM\u00b9.pkl",
        "our_data/LPT\u00b3.pkl",
    ],
)
def test_rejects_windows_reserved_member_name_variants(tmp_path: Path, member: str) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {member: b"cell"})

    with pytest.raises(ValueError, match="Windows reserved name"):
        audit_hust_archive(path, _manifest(path), _source())


@pytest.mark.parametrize(
    "names",
    [("our_data/a.pkl", "our_data/a.pkl"), ("our_data/A.pkl", "our_data/a.pkl")],
)
def test_rejects_duplicate_or_casefold_colliding_names(
    tmp_path: Path, names: tuple[str, str]
) -> None:
    path = tmp_path / "our_data.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("our_data/", b"")
            archive.writestr(names[0], b"one")
            archive.writestr(names[1], b"two")

    with pytest.raises(ValueError, match=r"duplicate|collision"):
        audit_hust_archive(path, _manifest(path), _source())


def test_rejects_symlink_member(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    link = zipfile.ZipInfo("our_data/link.pkl")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("our_data/", b"")
        archive.writestr(link, b"target")

    with pytest.raises(ValueError, match="regular file"):
        audit_hust_archive(path, _manifest(path), _source())


def test_rejects_symlink_disguised_as_root_directory(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    link = zipfile.ZipInfo("our_data/")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(link, b"")
        archive.writestr("our_data/1-1.pkl", b"cell")

    with pytest.raises(ValueError, match="unsafe file type"):
        audit_hust_archive(path, _manifest(path), _source())


def test_rejects_unapproved_compression_method(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"}, compression=zipfile.ZIP_BZIP2)

    with pytest.raises(ValueError, match="compression method"):
        audit_hust_archive(path, _manifest(path), _source())


def test_rejects_declared_zip_bomb_ratio_and_stream_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"0" * 4096})

    with pytest.raises(ValueError, match="compression ratio"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_compression_ratio=2.0),
        )

    with pytest.raises(ValueError, match="member size"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            limits=HustArchiveLimits(max_member_uncompressed_size=1024),
        )


@pytest.mark.parametrize("case", ["unknown", "missing", "hash"])
def test_rejects_inventory_mismatch(tmp_path: Path, case: str) -> None:
    path = tmp_path / "our_data.zip"
    payloads = {"our_data/1-1.pkl": b"one", "our_data/2-1.pkl": b"two"}
    _write_archive(path, payloads)
    expected = [
        ExpectedHustMember(path=name, sha256=hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    ]
    if case == "unknown":
        expected[1] = ExpectedHustMember(path="our_data/3-1.pkl", sha256=expected[1].sha256)
    elif case == "missing":
        expected.pop()
    else:
        expected[0] = expected[0].model_copy(update={"sha256": "0" * 64})

    with pytest.raises(ValueError, match="inventory"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            expected_inventory=tuple(expected),
            inventory_version="v1",
        )


def test_enforced_inventory_requires_a_nonempty_version(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    payload = b"cell"
    _write_archive(path, {"our_data/1-1.pkl": payload})
    expected = (
        ExpectedHustMember(
            path="our_data/1-1.pkl", sha256=hashlib.sha256(payload).hexdigest()
        ),
    )

    with pytest.raises(ValueError, match="inventory_version"):
        audit_hust_archive(
            path,
            _manifest(path),
            _source(),
            expected_inventory=expected,
        )


def test_archive_audit_rejects_naive_audited_at(tmp_path: Path) -> None:
    path = tmp_path / "our_data.zip"
    _write_archive(path, {"our_data/1-1.pkl": b"cell"})
    audit = audit_hust_archive(path, _manifest(path), _source())

    with pytest.raises(ValueError, match="audited_at must include a timezone"):
        audit.model_validate({**audit.model_dump(), "audited_at": datetime(2026, 7, 12)})
