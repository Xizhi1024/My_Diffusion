import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SAMPLE_ID_REGEX = r"^(?P<patient_id>\d{3})(?P<slice_id>\d{3})$"

_DICOM_EXTENSIONS = {".dcm", ".dicom"}

_dicom_scan_errors: List[str] = []
"""Populated by ``build_sample_records``, consumed by ``write_report``."""

# ─── data models ────────────────────────────────────────────────────────


@dataclass
class DicomSlice:
    path: str
    sop_instance_uid: str = ""
    series_uid: str = ""
    instance_number: str = ""
    z: str = ""


@dataclass
class SampleRecord:
    sample_id: str
    patient_id: str
    slice_id: int
    split: str

    ct_png_path: str = ""
    pet_png_path: str = ""
    ct_label_png_path: str = ""
    pet_label_png_path: str = ""
    ct_dicom_path: str = ""
    pet_dicom_path: str = ""
    ct_label_dicom_path: str = ""
    pet_label_dicom_path: str = ""

    has_ct_png: bool = False
    has_pet_png: bool = False
    has_ct_label_png: bool = False
    has_pet_label_png: bool = False
    has_ct_dicom: bool = False
    has_pet_dicom: bool = False
    has_ct_label_dicom: bool = False
    has_pet_label_dicom: bool = False

    ct_png_source_subset: str = ""
    pet_png_source_subset: str = ""
    label_source_subset: str = ""
    pet_label_source_subset: str = ""

    ct_dicom_sop_instance_uid: str = ""
    ct_dicom_series_uid: str = ""
    ct_dicom_instance_number: str = ""
    ct_dicom_z: str = ""

    pet_dicom_sop_instance_uid: str = ""
    pet_dicom_series_uid: str = ""
    pet_dicom_instance_number: str = ""
    pet_dicom_z: str = ""

    ct_label_dicom_sop_instance_uid: str = ""
    ct_label_dicom_series_uid: str = ""
    ct_label_dicom_instance_number: str = ""
    ct_label_dicom_z: str = ""

    pet_label_dicom_sop_instance_uid: str = ""
    pet_label_dicom_series_uid: str = ""
    pet_label_dicom_instance_number: str = ""
    pet_label_dicom_z: str = ""

    mapping_status: str = "ok"
    mapping_warning: str = ""


_CSV_COLUMNS = [
    "sample_id", "patient_id", "slice_id", "split",
    "ct_png_path", "pet_png_path", "ct_label_png_path", "pet_label_png_path",
    "ct_dicom_path", "pet_dicom_path", "ct_label_dicom_path", "pet_label_dicom_path",
    "has_ct_png", "has_pet_png", "has_ct_label_png", "has_pet_label_png",
    "has_ct_dicom", "has_pet_dicom", "has_ct_label_dicom", "has_pet_label_dicom",
    "ct_png_source_subset", "pet_png_source_subset",
    "label_source_subset", "pet_label_source_subset",
    "ct_dicom_sop_instance_uid", "ct_dicom_series_uid",
    "ct_dicom_instance_number", "ct_dicom_z",
    "pet_dicom_sop_instance_uid", "pet_dicom_series_uid",
    "pet_dicom_instance_number", "pet_dicom_z",
    "ct_label_dicom_sop_instance_uid", "ct_label_dicom_series_uid",
    "ct_label_dicom_instance_number", "ct_label_dicom_z",
    "pet_label_dicom_sop_instance_uid", "pet_label_dicom_series_uid",
    "pet_label_dicom_instance_number", "pet_label_dicom_z",
    "mapping_status", "mapping_warning",
]


# ─── helpers ────────────────────────────────────────────────────────────


def _basename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _set_status(rec: "SampleRecord", code: str) -> None:
    """Atomically set or append a machine-readable mapping status code.

    ``code`` must be one of: ``invalid_sample_id``, ``missing_ct_dicom``,
    ``missing_pet_dicom``, ``missing_ct_label_dicom``.
    Multiple codes are joined with ``;``.
    """
    if rec.mapping_status == "ok":
        rec.mapping_status = code
    elif code not in rec.mapping_status.split(";"):
        rec.mapping_status += ";" + code


def _append_detail(rec: "SampleRecord", msg: str) -> None:
    """Append human-readable context to ``mapping_warning``."""
    if rec.mapping_warning:
        rec.mapping_warning += "; "
    rec.mapping_warning += msg


# ─── sample-id parsing ──────────────────────────────────────────────────


def parse_sample_id(
    sample_id: str,
    pattern: str = DEFAULT_SAMPLE_ID_REGEX,
) -> Tuple[str, int]:
    """Parse a sample-id string into ``(patient_id, slice_id)``.

    Raises ``ValueError`` when *pattern* does not match.
    """
    m = re.match(pattern, sample_id)
    if not m:
        raise ValueError(
            f"sample_id {sample_id!r} does not match pattern {pattern!r}"
        )
    return m.group("patient_id"), int(m.group("slice_id"))


# ─── PNG scanning ───────────────────────────────────────────────────────


def scan_png_dir(root: Path) -> Dict[str, str]:
    """Return ``{sample_id: absolute_path}`` for every ``.png`` in *root*.

    Raises ``ValueError`` on duplicate *sample_id*.
    """
    if not root.is_dir():
        return {}

    mapping: Dict[str, str] = {}
    for p in sorted(root.glob("*.png")):
        sid = _basename_without_ext(str(p))
        if sid in mapping:
            raise ValueError(
                f"Duplicate sample_id {sid!r} in {root}: "
                f"{mapping[sid]} and {p}"
            )
        mapping[sid] = str(p.resolve())
    return mapping


# ─── DICOM scanning ─────────────────────────────────────────────────────


def _read_dicom_slice_meta(path: str) -> DicomSlice:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

    def _str_attr(name: str) -> str:
        val = getattr(ds, name, "")
        return str(val) if val else ""

    instance = _str_attr("InstanceNumber")

    z = ""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            z = str(ipp[2])
        except (TypeError, ValueError, IndexError):
            pass

    return DicomSlice(
        path=str(path),
        sop_instance_uid=_str_attr("SOPInstanceUID"),
        series_uid=_str_attr("SeriesInstanceUID"),
        instance_number=instance,
        z=z,
    )


def _sort_dicom_slices(
    slices: List[DicomSlice], order: str
) -> List[DicomSlice]:
    order = order.lower().strip()

    def _z_or_none(s: DicomSlice) -> Optional[float]:
        try:
            return float(s.z)
        except (TypeError, ValueError):
            return None

    def _instance_or_none(s: DicomSlice) -> int:
        try:
            return int(s.instance_number)
        except (TypeError, ValueError):
            return 10 ** 9

    if order == "instance_asc":
        return sorted(
            slices,
            key=lambda s: (
                s.instance_number == "",
                _instance_or_none(s),
                os.path.basename(s.path),
            ),
        )

    if order == "z_asc":
        return sorted(
            slices,
            key=lambda s: (
                _z_or_none(s) is None,
                _z_or_none(s) if _z_or_none(s) is not None else 1e18,
                _instance_or_none(s),
                os.path.basename(s.path),
            ),
        )

    if order == "filename":
        return sorted(slices, key=lambda s: os.path.basename(s.path))

    # default: z_desc
    return sorted(
        slices,
        key=lambda s: (
            _z_or_none(s) is None,
            -(_z_or_none(s) if _z_or_none(s) is not None else -1e18),
            _instance_or_none(s),
            os.path.basename(s.path),
        ),
    )


def scan_dicom_series(
    root: Path, order: str
) -> Tuple[Dict[str, List[DicomSlice]], List[str]]:
    """Return ``({patient_id: sorted_dicom_slices}, errors)``.

    *root* is expected to contain patient-id subdirectories, each holding
    ``.dcm`` / ``.dicom`` files.  Individual corrupt files are skipped
    (reported in *errors*) rather than aborting the whole scan.
    """
    if not root.is_dir():
        return {}, []

    errors: List[str] = []
    patient_map: Dict[str, List[DicomSlice]] = {}

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name

        slices: List[DicomSlice] = []
        for entry in sorted(patient_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in _DICOM_EXTENSIONS:
                continue
            try:
                sl = _read_dicom_slice_meta(str(entry))
                if not sl.sop_instance_uid and not sl.instance_number:
                    raise ValueError("empty or truncated DICOM (no usable metadata)")
                slices.append(sl)
            except Exception as exc:
                msg = f"Bad DICOM {entry}: {exc}"
                errors.append(msg)
                print(f"[data_manifest] WARNING: {msg}")

        if slices:
            patient_map[pid] = _sort_dicom_slices(slices, order=order)

    return patient_map, errors


# ─── record building ────────────────────────────────────────────────────


def _merge_png_scans(*scans: Dict[str, str]) -> Dict[str, str]:
    """Merge per-directory scans; duplicate sample_ids raise."""
    merged: Dict[str, str] = {}
    for scan in scans:
        for sid, path in scan.items():
            if sid in merged:
                raise ValueError(
                    f"Duplicate sample_id {sid!r} across PNG sources: "
                    f"{merged[sid]} and {path}"
                )
            merged[sid] = path
    return merged


def _build_slice_local_index(
    sample_ids: set,
    sample_id_regex: str,
) -> Dict[str, Dict[int, int]]:
    """Return ``{patient_id: {slice_id: local_index}}``.

    For each patient, available (parsed) slice-ids are sorted and assigned
    a 0-based local index.  Sample-ids that cannot be parsed are skipped.
    """
    patient_sids: Dict[str, List[int]] = {}
    for sid in sample_ids:
        try:
            pid, slice_idx = parse_sample_id(sid, pattern=sample_id_regex)
        except ValueError:
            continue
        patient_sids.setdefault(pid, []).append(slice_idx)

    result: Dict[str, Dict[int, int]] = {}
    for pid, sids in patient_sids.items():
        unique_sorted = sorted(set(sids))
        result[pid] = {sid: i for i, sid in enumerate(unique_sorted)}
    return result


def _attach_dicom_for_record(
    rec: "SampleRecord",
    patient_id: str,
    ct_local_idx: Optional[int],
    pet_local_idx: Optional[int],
    ct_label_local_idx: Optional[int],
    pet_label_local_idx: Optional[int],
    dicom_index_offset: int,
    ct_dcm_idx: Dict[str, List[DicomSlice]],
    pet_dcm_idx: Dict[str, List[DicomSlice]],
    ct_label_dcm_idx: Dict[str, List[DicomSlice]],
    pet_label_dcm_idx: Dict[str, List[DicomSlice]],
) -> None:
    """Attach DICOM paths/metadata using per-modality local-index mapping.

    Each modality uses its own local index, built solely from that
    modality's PNG source.  This prevents CT-only or PET-only slices from
    shifting the index of other modalities.
    """

    modality_configs = [
        ("CT", ct_dcm_idx, "ct_dicom_path", "has_ct_dicom", "ct_dicom", ct_local_idx),
        ("PET", pet_dcm_idx, "pet_dicom_path", "has_pet_dicom", "pet_dicom", pet_local_idx),
        (
            "CT_label", ct_label_dcm_idx, "ct_label_dicom_path",
            "has_ct_label_dicom", "ct_label_dicom", ct_label_local_idx,
        ),
        (
            "PET_label", pet_label_dcm_idx, "pet_label_dicom_path",
            "has_pet_label_dicom", "pet_label_dicom", pet_label_local_idx,
        ),
    ]

    for modality, dcm_map, path_attr, has_attr, prefix, local_idx in modality_configs:
        if local_idx is None:
            continue

        dcm_idx = local_idx + dicom_index_offset
        status_code = f"missing_{prefix}"

        if patient_id not in dcm_map:
            _set_status(rec, status_code)
            _append_detail(rec, f"{modality} DICOM patient {patient_id} not found")
            continue

        slices = dcm_map[patient_id]
        if 0 <= dcm_idx < len(slices):
            sl = slices[dcm_idx]
            setattr(rec, path_attr, sl.path)
            setattr(rec, has_attr, True)
            setattr(rec, f"{prefix}_sop_instance_uid", sl.sop_instance_uid)
            setattr(rec, f"{prefix}_series_uid", sl.series_uid)
            setattr(rec, f"{prefix}_instance_number", sl.instance_number)
            setattr(rec, f"{prefix}_z", sl.z)
        else:
            _set_status(rec, status_code)
            _append_detail(
                rec,
                f"{modality} DICOM index {dcm_idx} out of range "
                f"(patient {patient_id}, {len(slices)} slices)",
            )


def build_sample_records(
    raw_root: Path,
    dicom_root: Path,
    test_png_root: Optional[Path],
    sample_id_regex: str,
    dicom_slice_order: str,
    dicom_index_offset: int,
) -> List[SampleRecord]:
    """Scan CT/PET/label PNG and DICOM, producing one ``SampleRecord`` per
    unique sample-id encountered across all PNG sources.

    ``raw_root`` must contain ``part_CT/`` and ``part_PET/`` with
    ``train_data/{ImageSet,LabelSet}/PNG`` sub-directories.

    ``test_png_root`` is used to locate test-data PNGs
    (``<test_png_root>/part_CT/test_data/...``).  If ``test_png_root``
    is ``None``, *raw_root* is used for test data as well.

    ``dicom_root`` is expected to mirror the same tree with ``DICOM``
    directories in place of ``PNG``."""
    # pylint: disable=too-many-locals

    # ── scan PNG ────────────────────────────────────────────────────
    _test_root = test_png_root if test_png_root is not None else raw_root

    ct_png_train = scan_png_dir(
        raw_root / "part_CT" / "train_data" / "ImageSet" / "PNG",
    )
    ct_png_test = scan_png_dir(
        _test_root / "part_CT" / "test_data" / "ImageSet" / "PNG",
    )
    pet_png_train = scan_png_dir(
        raw_root / "part_PET" / "train_data" / "ImageSet" / "PNG",
    )
    pet_png_test = scan_png_dir(
        _test_root / "part_PET" / "test_data" / "ImageSet" / "PNG",
    )
    ct_label_train = scan_png_dir(
        raw_root / "part_CT" / "train_data" / "LabelSet" / "PNG",
    )
    ct_label_test = scan_png_dir(
        _test_root / "part_CT" / "test_data" / "LabelSet" / "PNG",
    )
    pet_label_train = scan_png_dir(
        raw_root / "part_PET" / "train_data" / "LabelSet" / "PNG",
    )
    pet_label_test = scan_png_dir(
        _test_root / "part_PET" / "test_data" / "LabelSet" / "PNG",
    )

    # ── validate critical directories ───────────────────────────────
    _required_dirs = [
        ("CT train PNG",     raw_root / "part_CT" / "train_data" / "ImageSet" / "PNG"),
        ("CT test  PNG",     _test_root / "part_CT" / "test_data" / "ImageSet" / "PNG"),
        ("PET train PNG",    raw_root / "part_PET" / "train_data" / "ImageSet" / "PNG"),
        ("PET test  PNG",    _test_root / "part_PET" / "test_data" / "ImageSet" / "PNG"),
        ("CT train label",   raw_root / "part_CT" / "train_data" / "LabelSet" / "PNG"),
        ("CT test  label",   _test_root / "part_CT" / "test_data" / "LabelSet" / "PNG"),
        ("PET train label",  raw_root / "part_PET" / "train_data" / "LabelSet" / "PNG"),
        ("PET test  label",  _test_root / "part_PET" / "test_data" / "LabelSet" / "PNG"),
    ]
    _missing: List[str] = []
    for _label, _dir in _required_dirs:
        if not _dir.is_dir():
            _missing.append(f"{_label}: {_dir}")
    if _missing:
        raise FileNotFoundError(
            "Critical data directories are missing:\n  "
            + "\n  ".join(_missing)
        )

    ct_png_all = _merge_png_scans(ct_png_train, ct_png_test)
    pet_png_all = _merge_png_scans(pet_png_train, pet_png_test)
    ct_label_all = _merge_png_scans(ct_label_train, ct_label_test)
    pet_label_all = _merge_png_scans(pet_label_train, pet_label_test)

    # ── scan DICOM ──────────────────────────────────────────────────
    ct_dcm_train, ct_dcm_train_err = scan_dicom_series(
        dicom_root / "part_CT" / "train_data" / "ImageSet" / "DICOM",
        order=dicom_slice_order,
    )
    ct_dcm_test, ct_dcm_test_err = scan_dicom_series(
        dicom_root / "part_CT" / "test_data" / "ImageSet" / "DICOM",
        order=dicom_slice_order,
    )
    _dup = set(ct_dcm_train) & set(ct_dcm_test)
    if _dup:
        raise ValueError(
            f"CT DICOM patient overlap between train and test: {sorted(_dup)}"
        )
    ct_dcm_idx = {**ct_dcm_train, **ct_dcm_test}

    pet_dcm_train, pet_dcm_train_err = scan_dicom_series(
        dicom_root / "part_PET" / "train_data" / "ImageSet" / "DICOM",
        order=dicom_slice_order,
    )
    pet_dcm_test, pet_dcm_test_err = scan_dicom_series(
        dicom_root / "part_PET" / "test_data" / "ImageSet" / "DICOM",
        order=dicom_slice_order,
    )
    _dup = set(pet_dcm_train) & set(pet_dcm_test)
    if _dup:
        raise ValueError(
            f"PET DICOM patient overlap between train and test: {sorted(_dup)}"
        )
    pet_dcm_idx = {**pet_dcm_train, **pet_dcm_test}

    ct_label_dcm_train, ct_label_dcm_train_err = scan_dicom_series(
        dicom_root / "part_CT" / "train_data" / "LabelSet" / "DICOM",
        order=dicom_slice_order,
    )
    ct_label_dcm_test, ct_label_dcm_test_err = scan_dicom_series(
        dicom_root / "part_CT" / "test_data" / "LabelSet" / "DICOM",
        order=dicom_slice_order,
    )
    _dup = set(ct_label_dcm_train) & set(ct_label_dcm_test)
    if _dup:
        raise ValueError(
            f"CT label DICOM patient overlap between train and test: {sorted(_dup)}"
        )
    ct_label_dcm_idx = {**ct_label_dcm_train, **ct_label_dcm_test}

    pet_label_dcm_train, pet_label_dcm_train_err = scan_dicom_series(
        dicom_root / "part_PET" / "train_data" / "LabelSet" / "DICOM",
        order=dicom_slice_order,
    )
    pet_label_dcm_test, pet_label_dcm_test_err = scan_dicom_series(
        dicom_root / "part_PET" / "test_data" / "LabelSet" / "DICOM",
        order=dicom_slice_order,
    )
    _dup = set(pet_label_dcm_train) & set(pet_label_dcm_test)
    if _dup:
        raise ValueError(
            f"PET label DICOM patient overlap between train and test: {sorted(_dup)}"
        )
    pet_label_dcm_idx = {**pet_label_dcm_train, **pet_label_dcm_test}

    all_dcm_errors = (
        ct_dcm_train_err + ct_dcm_test_err
        + pet_dcm_train_err + pet_dcm_test_err
        + ct_label_dcm_train_err + ct_label_dcm_test_err
        + pet_label_dcm_train_err + pet_label_dcm_test_err
    )
    global _dicom_scan_errors
    _dicom_scan_errors = all_dcm_errors
    if all_dcm_errors:
        print(
            f"[data_manifest] WARNING: {len(all_dcm_errors)} DICOM file(s) "
            "skipped (see errors above)"
        )

    # ── per-modality slice_id -> local_index ────────────────────────
    # Each modality's index is built ONLY from its own PNG samples so that
    # CT-only / PET-only / label-only slices cannot shift another modality.
    ct_sid_to_local = _build_slice_local_index(
        set(ct_png_all), sample_id_regex,
    )
    pet_sid_to_local = _build_slice_local_index(
        set(pet_png_all), sample_id_regex,
    )
    label_sid_to_local = _build_slice_local_index(
        set(ct_label_all), sample_id_regex,
    )
    pet_label_sid_to_local = _build_slice_local_index(
        set(pet_label_all), sample_id_regex,
    )

    # ── build records ───────────────────────────────────────────────
    all_sample_ids = (
        set(ct_png_all) | set(pet_png_all)
        | set(ct_label_all) | set(pet_label_all)
    )
    records: List[SampleRecord] = []

    for sid in sorted(all_sample_ids):
        # parse id
        try:
            patient_id, slice_idx = parse_sample_id(sid, pattern=sample_id_regex)
        except ValueError:
            # Mark as invalid immediately, keep sample
            rec = SampleRecord(
                sample_id=sid,
                patient_id=sid,
                slice_id=0,
                split="",
                mapping_status="invalid_sample_id",
                mapping_warning=f"sample_id {sid!r} does not match pattern",
            )
            # Still record PNG presence and subset source if any
            if sid in ct_png_all:
                rec.ct_png_path = ct_png_all[sid]
                rec.has_ct_png = True
                rec.ct_png_source_subset = (
                    "test" if sid in ct_png_test else "train"
                )
            if sid in pet_png_all:
                rec.pet_png_path = pet_png_all[sid]
                rec.has_pet_png = True
                rec.pet_png_source_subset = (
                    "test" if sid in pet_png_test else "train"
                )
            if sid in ct_label_all:
                rec.ct_label_png_path = ct_label_all[sid]
                rec.has_ct_label_png = True
                rec.label_source_subset = (
                    "test" if sid in ct_label_test else "train"
                )
            if sid in pet_label_all:
                rec.pet_label_png_path = pet_label_all[sid]
                rec.has_pet_label_png = True
                rec.pet_label_source_subset = (
                    "test" if sid in pet_label_test else "train"
                )
            records.append(rec)
            continue

        rec = SampleRecord(
            sample_id=sid,
            patient_id=patient_id,
            slice_id=slice_idx,
            split="",
        )

        # PNG paths
        if sid in ct_png_all:
            rec.ct_png_path = ct_png_all[sid]
            rec.has_ct_png = True
            rec.ct_png_source_subset = (
                "test" if sid in ct_png_test else "train"
            )
        if sid in pet_png_all:
            rec.pet_png_path = pet_png_all[sid]
            rec.has_pet_png = True
            rec.pet_png_source_subset = (
                "test" if sid in pet_png_test else "train"
            )
        if sid in ct_label_all:
            rec.ct_label_png_path = ct_label_all[sid]
            rec.has_ct_label_png = True
            rec.label_source_subset = (
                "test" if sid in ct_label_test else "train"
            )
        if sid in pet_label_all:
            rec.pet_label_png_path = pet_label_all[sid]
            rec.has_pet_label_png = True
            rec.pet_label_source_subset = (
                "test" if sid in pet_label_test else "train"
            )

        # DICOM matching: each modality uses its own per-patient local index
        ct_local_idx = ct_sid_to_local.get(patient_id, {}).get(slice_idx)
        pet_local_idx = pet_sid_to_local.get(patient_id, {}).get(slice_idx)
        label_local_idx = label_sid_to_local.get(patient_id, {}).get(slice_idx)
        pet_label_local_idx = pet_label_sid_to_local.get(patient_id, {}).get(slice_idx)

        _attach_dicom_for_record(
            rec,
            patient_id=patient_id,
            ct_local_idx=ct_local_idx,
            pet_local_idx=pet_local_idx,
            ct_label_local_idx=label_local_idx,
            pet_label_local_idx=pet_label_local_idx,
            dicom_index_offset=dicom_index_offset,
            ct_dcm_idx=ct_dcm_idx,
            pet_dcm_idx=pet_dcm_idx,
            ct_label_dcm_idx=ct_label_dcm_idx,
            pet_label_dcm_idx=pet_label_dcm_idx,
        )

        records.append(rec)

    return records


# ─── split assignment ───────────────────────────────────────────────────


def assign_splits(
    records: List[SampleRecord],
    test_sample_ids: Optional[set] = None,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    """Assign ``split`` on every record.

    Test patients are identified by (in order):

    1. Explicit *test_sample_ids* (if non-empty).
    2. Fallback: any record whose ``*_source_subset`` field equals ``"test"``
       marks its patient for the test split.

    Remaining patients are shuffled (using *seed*) and split by
    *val_ratio* into ``"train"`` / ``"val"``.
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    import random as _random

    test_patients: set = set()

    # 1) explicit set
    if test_sample_ids:
        for rec in records:
            if rec.sample_id in test_sample_ids:
                test_patients.add(rec.patient_id)

    # 2) fallback: source_subset markers
    if not test_patients:
        _test_source_fields = (
            "ct_png_source_subset", "pet_png_source_subset",
            "label_source_subset", "pet_label_source_subset",
        )
        for rec in records:
            for _f in _test_source_fields:
                if getattr(rec, _f, "") == "test":
                    test_patients.add(rec.patient_id)
                    break

    for rec in records:
        if rec.patient_id in test_patients:
            rec.split = "test"

    remaining_patients = sorted(
        {rec.patient_id for rec in records if rec.split != "test"}
    )

    rng = _random.Random(seed)
    rng.shuffle(remaining_patients)

    n_val = int(len(remaining_patients) * val_ratio)
    val_patients = set(remaining_patients[:n_val])
    train_patients = set(remaining_patients[n_val:])

    for rec in records:
        if rec.split:
            continue
        if rec.patient_id in val_patients:
            rec.split = "val"
        else:
            rec.split = "train"

    _verify_split_no_overlap(records)


def _verify_split_no_overlap(records: List[SampleRecord]) -> None:
    train_p = {rec.patient_id for rec in records if rec.split == "train"}
    val_p = {rec.patient_id for rec in records if rec.split == "val"}
    test_p = {rec.patient_id for rec in records if rec.split == "test"}

    if train_p & val_p:
        raise ValueError(f"train/val patient overlap: {train_p & val_p}")
    if train_p & test_p:
        raise ValueError(f"train/test patient overlap: {train_p & test_p}")
    if val_p & test_p:
        raise ValueError(f"val/test patient overlap: {val_p & test_p}")


# ─── fingerprint ────────────────────────────────────────────────────────

_FINGERPRINT_FIELDS = (
    "sample_id", "patient_id", "slice_id", "split",
    "ct_png_path", "pet_png_path", "ct_label_png_path", "pet_label_png_path",
    "ct_dicom_path", "pet_dicom_path", "ct_label_dicom_path", "pet_label_dicom_path",
    "ct_dicom_sop_instance_uid", "ct_dicom_series_uid",
    "pet_dicom_sop_instance_uid", "pet_dicom_series_uid",
    "ct_label_dicom_sop_instance_uid", "ct_label_dicom_series_uid",
    "pet_label_dicom_sop_instance_uid", "pet_label_dicom_series_uid",
    "mapping_status",
)


def _fingerprint_value(rec: SampleRecord, field: str) -> str:
    val = getattr(rec, field, "")
    if not val:
        return ""
    if field.endswith("_path"):
        basename = os.path.basename(val)
        try:
            fsize = os.path.getsize(val)
        except OSError:
            fsize = 0
        return f"{basename}:{fsize}"
    return val


def compute_dataset_fingerprint(records: List[SampleRecord]) -> str:
    """Stable sha256 hex digest (first 12 chars) over ordered records.

    Paths are reduced to ``basename:filesize`` so the fingerprint is
    portable across machines yet sensitive to content changes.
    Also includes split, DICOM UIDs and mapping_status so that different
    train/val seeds or changed DICOM links produce distinct fingerprints.
    """
    keys = sorted(
        tuple(_fingerprint_value(r, f) for f in _FINGERPRINT_FIELDS)
        for r in records
    )
    payload = json.dumps(keys, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ─── serialisation ──────────────────────────────────────────────────────


def _record_to_row(rec: SampleRecord) -> Dict[str, Any]:
    d = asdict(rec)
    return {k: d.get(k, "") for k in _CSV_COLUMNS}


def write_csv(records: List[SampleRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(_record_to_row(rec))


def write_jsonl(records: List[SampleRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_record_to_row(rec), ensure_ascii=False) + "\n")


def write_split_csvs(records: List[SampleRecord], splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        subset = [r for r in records if r.split == split_name]
        write_csv(subset, splits_dir / f"{split_name}.csv")


# ─── report ─────────────────────────────────────────────────────────────


def _status_has(records: List[SampleRecord], code: str) -> List[str]:
    """Return sample_ids whose mapping_status contains *code*."""
    return sorted(
        r.sample_id for r in records if code in r.mapping_status.split(";")
    )


def write_report(
    records: List[SampleRecord],
    out_root: Path,
    dicom_errors: Optional[List[str]] = None,
) -> None:
    """Write ``report.json`` and ``report.md`` to *out_root*.

    *dicom_errors* (when provided) are included in the report.  When
    omitted, falls back to the module-level ``_dicom_scan_errors``
    populated by the most recent ``build_sample_records`` call."""
    _errs = dicom_errors if dicom_errors is not None else _dicom_scan_errors
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(records)
    paired_ct_pet = sum(1 for r in records if r.has_ct_png and r.has_pet_png)
    ct_label_count = sum(1 for r in records if r.has_ct_label_png)
    pet_label_count = sum(1 for r in records if r.has_pet_label_png)
    trains = [r for r in records if r.split == "train"]
    vals = [r for r in records if r.split == "val"]
    tests = [r for r in records if r.split == "test"]

    missing_ct = sorted(r.sample_id for r in records if not r.has_ct_png)
    missing_pet = sorted(r.sample_id for r in records if not r.has_pet_png)
    missing_label = sorted(
        r.sample_id for r in records if not r.has_ct_label_png
    )
    missing_pet_label = sorted(
        r.sample_id for r in records if not r.has_pet_label_png
    )
    missing_ct_dcm = sorted(
        r.sample_id for r in records
        if r.has_ct_png and not r.has_ct_dicom
    )
    missing_pet_dcm = sorted(
        r.sample_id for r in records
        if r.has_pet_png and not r.has_pet_dicom
    )
    missing_label_dcm = sorted(
        r.sample_id for r in records
        if r.has_ct_label_png and not r.has_ct_label_dicom
    )
    missing_pet_label_dcm = sorted(
        r.sample_id for r in records
        if r.has_pet_label_png and not r.has_pet_label_dicom
    )
    invalid_ids = _status_has(records, "invalid_sample_id")
    mapping_issues = sorted(
        r.sample_id
        for r in records
        if r.mapping_status not in ("", "ok")
    )
    test_source_count = sum(
        1
        for r in tests
        if "test" in (r.ct_png_source_subset, r.pet_png_source_subset)
    )

    fingerprint = compute_dataset_fingerprint(records)

    train_patients = sorted({r.patient_id for r in trains})
    val_patients = sorted({r.patient_id for r in vals})
    test_patients = sorted({r.patient_id for r in tests})

    overlap_checks = {
        "train_val": bool(set(train_patients) & set(val_patients)),
        "train_test": bool(set(train_patients) & set(test_patients)),
        "val_test": bool(set(val_patients) & set(test_patients)),
    }

    report = {
        "total_records": total,
        "paired_ct_pet_png_count": paired_ct_pet,
        "ct_label_png_count": ct_label_count,
        "pet_label_png_count": pet_label_count,
        "train_record_count": len(trains),
        "val_record_count": len(vals),
        "test_record_count": len(tests),
        "train_patient_count": len(train_patients),
        "val_patient_count": len(val_patients),
        "test_patient_count": len(test_patients),
        "missing_ct_png": missing_ct,
        "missing_pet_png": missing_pet,
        "missing_ct_label_png": missing_label,
        "missing_pet_label_png": missing_pet_label,
        "missing_ct_dicom": missing_ct_dcm,
        "missing_pet_dicom": missing_pet_dcm,
        "missing_ct_label_dicom": missing_label_dcm,
        "missing_pet_label_dicom": missing_pet_label_dcm,
        "invalid_sample_id": invalid_ids,
        "mapping_issues": mapping_issues,
        "patient_overlap_check": overlap_checks,
        "test_source_count": test_source_count,
        "dataset_fingerprint": fingerprint,
        "dicom_scan_errors_count": len(_errs),
        "dicom_scan_errors": _errs[:50],
    }

    with open(out_root / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    md_lines = [
        "# Dataset Manifest Audit Report",
        "",
        f"- **Total records**: {total}",
        f"- **Paired CT+PET PNG**: {paired_ct_pet}",
        f"- **CT label PNG**: {ct_label_count}",
        f"- **PET label PNG**: {pet_label_count}",
        f"- **Dataset fingerprint**: `{fingerprint}`",
        "",
        "## Split Summary",
        "",
        f"| Split | Records | Patients |",
        f"|-------|---------|----------|",
        f"| train | {len(trains)} | {len(train_patients)} |",
        f"| val   | {len(vals)} | {len(val_patients)} |",
        f"| test  | {len(tests)} | {len(test_patients)} |",
        "",
        f"- **test samples from test_data directory**: {test_source_count}",
        "",
        "## Patient Overlap Check",
        "",
        f"- train/val overlap: {'**FAIL**' if overlap_checks['train_val'] else 'OK'}",
        f"- train/test overlap: {'**FAIL**' if overlap_checks['train_test'] else 'OK'}",
        f"- val/test overlap: {'**FAIL**' if overlap_checks['val_test'] else 'OK'}",
        "",
        "## Completeness",
        "",
        f"- Missing CT PNG ({len(missing_ct)}): {_fmt_list(missing_ct)}",
        f"- Missing PET PNG ({len(missing_pet)}): {_fmt_list(missing_pet)}",
        f"- Missing CT label PNG ({len(missing_label)}): {_fmt_list(missing_label)}",
        f"- Missing PET label PNG ({len(missing_pet_label)}): {_fmt_list(missing_pet_label)}",
        f"- Missing CT DICOM ({len(missing_ct_dcm)}): {_fmt_list(missing_ct_dcm)}",
        f"- Missing PET DICOM ({len(missing_pet_dcm)}): {_fmt_list(missing_pet_dcm)}",
        f"- Missing CT label DICOM ({len(missing_label_dcm)}): {_fmt_list(missing_label_dcm)}",
        f"- Missing PET label DICOM ({len(missing_pet_label_dcm)}): {_fmt_list(missing_pet_label_dcm)}",
        f"- Invalid sample IDs ({len(invalid_ids)}): {_fmt_list(invalid_ids)}",
        f"- Records with mapping issues ({len(mapping_issues)}): {_fmt_list(mapping_issues)}",
        f"- DICOM scan errors ({len(_errs)} total, showing first 10):",
    ]

    for _err in _errs[:10]:
        md_lines.append(f"  - {_err}")
    if len(_errs) > 10:
        md_lines.append(f"  - ... and {len(_errs) - 10} more")

    md_lines += [
        "",
        "## Unpaired Samples",
        "",
    ]

    unpaired = [r for r in records if r.has_ct_png != r.has_pet_png]
    if unpaired:
        for r in sorted(unpaired, key=lambda x: x.sample_id):
            md_lines.append(
                f"- `{r.sample_id}`: "
                f"CT={'Y' if r.has_ct_png else 'N'} "
                f"PET={'Y' if r.has_pet_png else 'N'}"
            )
    else:
        md_lines.append("(none)")

    md_lines.append("")
    md_lines.append(f"*Generated: {datetime.now().isoformat()}*")
    md_lines.append("")

    with open(out_root / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def _fmt_list(items: list, max_show: int = 20) -> str:
    if not items:
        return "(none)"
    if len(items) <= max_show:
        return ", ".join(f"`{x}`" for x in items)
    shown = ", ".join(f"`{x}`" for x in items[:max_show])
    return f"{shown}, ... ({len(items)} total)"
