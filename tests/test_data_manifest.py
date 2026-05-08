"""Tests for src/data_manifest.py --- uses ``tmp_path``, no real Data dependency."""

import json
from pathlib import Path

import pytest

from src.data.data_manifest import (
    DEFAULT_SAMPLE_ID_REGEX,
    DicomSlice,
    SampleRecord,
    assign_splits,
    build_sample_records,
    compute_dataset_fingerprint,
    parse_sample_id,
    scan_dicom_series,
    scan_png_dir,
    write_csv,
    write_jsonl,
    write_report,
    write_split_csvs,
)


# ── helpers ─────────────────────────────────────────────────────────────

def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_fake_raw(tmp_path: Path) -> Path:
    """Create a minimal raw-root mirroring the real Data layout."""
    raw = tmp_path / "Data"
    # subset_A train PNGs (samples: patient 004 slices 2-4, patient 005 slices 2-3)
    for sid in ("004002", "004003", "004004", "005002", "005003"):
        _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / f"{sid}.png")
        _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / f"{sid}.png")
    # subset_A label PNGs
    for sid in ("004002", "004003"):
        _touch(raw / "subset_A" / "part_CT" / "train_data" / "LabelSet" / "PNG" / f"{sid}.png")
    # subset_A test_data label
    _touch(raw / "subset_A" / "part_CT" / "test_data" / "label" / "001002.png")
    return raw


def _build_fake_test_root(tmp_path: Path) -> Path:
    """Create minimal subset_B."""
    root = tmp_path / "subset_B"
    for sid in ("001002", "001003", "002002"):
        _touch(root / "part_CT" / "train_data" / "ImageSet" / "PNG" / f"{sid}.png")
        _touch(root / "part_PET" / "ImageSet" / "PNG" / f"{sid}.png")
    return root


def _build_fake_dicom_root(tmp_path: Path) -> Path:
    """Create minimal DICOM directories with real .dcm files."""
    import pydicom
    from pydicom.dataset import Dataset

    root = tmp_path / "dicom_root"

    for modality, sub in (("CT", "part_CT/train_data/ImageSet/DICOM"),
                          ("PET", "part_PET/ImageSet/DICOM"),
                          ("CT_label", "part_CT/train_data/LabelSet/DICOM")):
        for pid in ("004", "005"):
            pdir = root / sub / pid
            pdir.mkdir(parents=True, exist_ok=True)
            pid_int = int(pid)
            for i in range(3):
                ds = Dataset()
                ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.{pid_int}.{i}"
                ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.9.7132.{pid_int}"
                ds.InstanceNumber = str(i + 1)
                ds.ImagePositionPatient = ["0.0", "0.0", str(float(3 - i))]
                ds.PatientID = pid
                ds.Modality = "CT" if "CT" in modality else "PT"
                pydicom.dcmwrite(
                    str(pdir / f"{i + 1}.dcm"), ds,
                    little_endian=True, implicit_vr=True,
                )
    return root


# ═══ 1. parse_sample_id ═════════════════════════════════════════════════

class TestParseSampleId:
    def test_normal(self):
        pid, sid = parse_sample_id("001002")
        assert pid == "001"
        assert sid == 2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_sample_id("abcd")

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            parse_sample_id("12345")

    def test_custom_pattern(self):
        pat = r"^(?P<patient_id>[A-Z]{2})(?P<slice_id>\d{2})$"
        pid, sid = parse_sample_id("AB12", pattern=pat)
        assert pid == "AB"
        assert sid == 12


# ═══ 2. scan_png_dir ════════════════════════════════════════════════════

class TestScanPngDir:
    def test_normal(self, tmp_path):
        d = tmp_path / "pngs"
        d.mkdir()
        _touch(d / "001002.png")
        _touch(d / "003004.png")
        result = scan_png_dir(d)
        assert len(result) == 2
        assert result["001002"].endswith("001002.png")
        assert result["003004"].endswith("003004.png")

    def test_duplicate_raises(self, tmp_path):
        from src.data.data_manifest import _merge_png_scans
        s1 = {"x": "/a/x.png"}
        s2 = {"x": "/b/x.png"}
        with pytest.raises(ValueError, match="Duplicate sample_id"):
            _merge_png_scans(s1, s2)

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert scan_png_dir(d) == {}

    def test_nonexistent_dir(self, tmp_path):
        assert scan_png_dir(tmp_path / "nope") == {}


# ═══ 3. subset_B test assignment ════════════════════════════════════════

class TestSubsetBTestAssignment:
    def test_subset_b_goes_to_test(self, tmp_path):
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        test_ids = {
            r.sample_id for r in records
            if "subset_B" in (r.ct_png_source_subset, r.pet_png_source_subset)
        }
        assign_splits(records, test_ids, val_ratio=0.15, seed=42)

        for r in records:
            if r.sample_id in test_ids:
                assert r.split == "test", f"{r.sample_id} should be test"
            # patient leakage prevention: patient 001, 002 appear in subset_B
            if r.patient_id in {"001", "002"}:
                assert r.split == "test", (
                    f"{r.sample_id} (patient {r.patient_id}) should be test"
                )


# ═══ 4. patient-level split no overlap ══════════════════════════════════

class TestSplitNoOverlap:
    def test_no_patient_overlap(self, tmp_path):
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        test_ids = {
            r.sample_id for r in records
            if "subset_B" in (r.ct_png_source_subset, r.pet_png_source_subset)
        }
        assign_splits(records, test_ids, val_ratio=0.2, seed=123)

        train_p = {r.patient_id for r in records if r.split == "train"}
        val_p = {r.patient_id for r in records if r.split == "val"}
        test_p = {r.patient_id for r in records if r.split == "test"}

        assert not (train_p & val_p)
        assert not (train_p & test_p)
        assert not (val_p & test_p)


# ═══ 5. unpaired retention ══════════════════════════════════════════════

class TestUnpairedRetention:
    def test_ct_only_not_dropped(self, tmp_path):
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / "099001.png")

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        rec = next(r for r in records if r.sample_id == "099001")
        assert rec.has_ct_png is True
        assert rec.has_pet_png is False

    def test_pet_only_not_dropped(self, tmp_path):
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / "088001.png")

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        rec = next(r for r in records if r.sample_id == "088001")
        assert rec.has_pet_png is True
        assert rec.has_ct_png is False


# ═══ 6. serialisation ═══════════════════════════════════════════════════

class TestSerialisation:
    @pytest.fixture
    def sample_records(self):
        return [
            SampleRecord(
                sample_id="001002", patient_id="001", slice_id=2, split="test",
                ct_png_path="/d/ct/001002.png", pet_png_path="/d/pet/001002.png",
                has_ct_png=True, has_pet_png=True,
            ),
            SampleRecord(
                sample_id="001003", patient_id="001", slice_id=3, split="test",
                ct_png_path="/d/ct/001003.png",
                has_ct_png=True, has_pet_png=False,
                mapping_status="missing_pet_dicom",
                mapping_warning="PET DICOM patient 001 not found",
            ),
        ]

    def test_write_csv(self, tmp_path, sample_records):
        p = tmp_path / "test.csv"
        write_csv(sample_records, p)
        content = p.read_text(encoding="utf-8")
        assert "sample_id" in content
        assert "001002" in content
        assert "missing_pet_dicom" in content

    def test_write_jsonl(self, tmp_path, sample_records):
        p = tmp_path / "test.jsonl"
        write_jsonl(sample_records, p)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        data = [json.loads(line) for line in lines]
        assert data[1]["mapping_status"] == "missing_pet_dicom"

    def test_write_split_csvs(self, tmp_path, sample_records):
        d = tmp_path / "splits"
        write_split_csvs(sample_records, d)
        assert (d / "test.csv").exists()

    def test_write_report(self, tmp_path, sample_records):
        write_report(sample_records, tmp_path)
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["total_records"] == 2
        assert report["test_record_count"] == 2


# ═══ 7. fingerprint ═════════════════════════════════════════════════════

class TestFingerprint:
    def test_same_records_same_fingerprint(self):
        r1 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="test",
            ct_png_path="/a.png", pet_png_path="/b.png",
            has_ct_png=True, has_pet_png=True,
        )
        r2 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="test",
            ct_png_path="/a.png", pet_png_path="/b.png",
            has_ct_png=True, has_pet_png=True,
        )
        assert compute_dataset_fingerprint([r1]) == compute_dataset_fingerprint([r2])

    def test_different_split_different_fingerprint(self):
        r1 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="train",
            ct_png_path="/a.png",
        )
        r2 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="val",
            ct_png_path="/a.png",
        )
        assert compute_dataset_fingerprint([r1]) != compute_dataset_fingerprint([r2])

    def test_different_status_different_fingerprint(self):
        r1 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="test",
            mapping_status="ok",
        )
        r2 = SampleRecord(
            sample_id="001002", patient_id="001", slice_id=2, split="test",
            mapping_status="missing_ct_dicom",
        )
        assert compute_dataset_fingerprint([r1]) != compute_dataset_fingerprint([r2])


# ═══ 8. DICOM scan ══════════════════════════════════════════════════════

class TestDicomScan:
    def test_scan_with_real_files(self, tmp_path):
        import pydicom
        from pydicom.dataset import Dataset

        dcm_root = tmp_path / "dicom"
        patient_dir = dcm_root / "001"
        patient_dir.mkdir(parents=True)

        for i in range(3):
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.1.{i}"
            ds.SeriesInstanceUID = "1.2.826.0.1.3680043.9.7132.1"
            ds.InstanceNumber = str(3 - i)
            ds.ImagePositionPatient = ["0.0", "0.0", str(float(i))]
            ds.PatientID = "001"
            ds.Modality = "CT"
            pydicom.dcmwrite(
                str(patient_dir / f"slice_{i}.dcm"), ds,
                little_endian=True, implicit_vr=True,
            )

        result, errors = scan_dicom_series(dcm_root, order="z_desc")
        assert errors == []
        assert "001" in result
        slices = result["001"]
        assert len(slices) == 3
        # z_desc: highest z first (i=2 -> z=2.0)
        assert slices[0].z == "2.0"

    def test_bad_file_skipped(self, tmp_path):
        dcm_root = tmp_path / "dicom"
        patient_dir = dcm_root / "001"
        patient_dir.mkdir(parents=True)
        # Write a truncated file: valid DICOM preamble but cut off mid-element
        preamble = b"\x00" * 128 + b"DICM"
        (patient_dir / "truncated.dcm").write_bytes(preamble + b"\x01\x02\x03")

        result, errors = scan_dicom_series(dcm_root, order="z_desc")
        assert len(errors) >= 1
        assert "truncated.dcm" in errors[0]
        assert result == {} or "001" not in result

    def test_missing_dir(self, tmp_path):
        result, errors = scan_dicom_series(tmp_path / "nope", order="z_desc")
        assert result == {}
        assert errors == []


# ═══ 9. mapping_status semantics ════════════════════════════════════════

class TestMappingStatus:
    def test_invalid_sample_id_marked_immediately(self, tmp_path):
        """Parse failure must set mapping_status='invalid_sample_id' regardless."""
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / "bad.png")
        _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / "bad.png")

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        bad = next(r for r in records if r.sample_id == "bad")
        assert "invalid_sample_id" in bad.mapping_status.split(";"), (
            f"Expected invalid_sample_id, got {bad.mapping_status!r}"
        )
        # Still records PNG presence
        assert bad.has_ct_png is True
        assert bad.has_pet_png is True

    def test_missing_dicom_produces_status_code(self, tmp_path):
        """When DICOM index is out of range, status must be a specific code."""
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        # sample 004004: patient 004, slice_id=4 → local_index=2 (0:2,1:3,2:4)
        # + offset 0 → dcm_idx=2 → valid (3 slices: idx 0,1,2)
        # sample 005003: patient 005, slice_id=3 → local_index=1
        # + offset 0 → dcm_idx=1 → valid
        # Add a sample that will be out of range:
        # sample 004006: patient 004, slice_id=6 → local_index=3 → dcm_idx=3 → out of range
        _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / "004006.png")
        _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / "004006.png")

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        rec = next(r for r in records if r.sample_id == "004006")
        assert not rec.has_ct_dicom
        assert "missing_ct_dicom" in rec.mapping_status.split(";"), (
            f"Expected missing_ct_dicom in status, got {rec.mapping_status!r}"
        )

    def test_ok_status_preserved_when_all_good(self, tmp_path):
        """When everything matches, mapping_status stays 'ok'."""
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        # 004002: patient 004, slice_id=2 → local_index=0 → dcm_idx=0 → valid
        rec = next(r for r in records if r.sample_id == "004002")
        assert rec.has_ct_dicom
        assert rec.has_pet_dicom
        assert rec.mapping_status == "ok", (
            f"Expected 'ok', got {rec.mapping_status!r}"
        )


# ═══ 10. DICOM mapping: per-patient local index ═════════════════════════

class TestDicommapping:
    def test_non_contiguous_slice_ids(self, tmp_path):
        """PNG slice_ids not starting from 001 must still map correctly."""
        raw = tmp_path / "Data"
        test_root = tmp_path / "subset_B"
        test_root.mkdir()
        (test_root / "part_CT" / "train_data" / "ImageSet" / "PNG").mkdir(parents=True)
        (test_root / "part_PET" / "ImageSet" / "PNG").mkdir(parents=True)

        import pydicom
        from pydicom.dataset import Dataset

        # Patient 007 has PNG slice_ids 010, 015, 020 (not starting at 001)
        for sid in ("007010", "007015", "007020"):
            _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / f"{sid}.png")
            _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / f"{sid}.png")

        # DICOM for patient 007 has 5 slices (indices 0-4)
        dcm_root = tmp_path / "dicom_root"
        pdir = dcm_root / "part_CT" / "train_data" / "ImageSet" / "DICOM" / "007"
        pdir.mkdir(parents=True)
        for i in range(5):
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.7.{i}"
            ds.SeriesInstanceUID = "1.2.826.0.1.3680043.9.7132.7"
            ds.InstanceNumber = str(i + 1)
            ds.ImagePositionPatient = ["0.0", "0.0", str(float(4 - i))]
            ds.PatientID = "007"
            ds.Modality = "CT"
            pydicom.dcmwrite(str(pdir / f"{i + 1}.dcm"), ds, little_endian=True, implicit_vr=True)
        # Same for PET
        pdir2 = dcm_root / "part_PET" / "ImageSet" / "DICOM" / "007"
        pdir2.mkdir(parents=True)
        for i in range(5):
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.7.{i}"
            ds.SeriesInstanceUID = "1.2.826.0.1.3680043.9.7132.7"
            ds.InstanceNumber = str(i + 1)
            ds.ImagePositionPatient = ["0.0", "0.0", str(float(4 - i))]
            ds.PatientID = "007"
            ds.Modality = "PT"
            pydicom.dcmwrite(str(pdir2 / f"{i + 1}.dcm"), ds, little_endian=True, implicit_vr=True)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm_root, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )

        # 007010: slice_id=10 → local_index=0 → dcm_idx=0 → first DICOM
        rec10 = next(r for r in records if r.sample_id == "007010")
        assert rec10.has_ct_dicom, f"007010 should have CT DICOM: {rec10.mapping_status}"
        assert rec10.ct_dicom_instance_number == "1"

        # 007015: slice_id=15 → local_index=1 → dcm_idx=1 → second DICOM
        rec15 = next(r for r in records if r.sample_id == "007015")
        assert rec15.has_ct_dicom
        assert rec15.ct_dicom_instance_number == "2"

        # 007020: slice_id=20 → local_index=2 → dcm_idx=2 → third DICOM
        rec20 = next(r for r in records if r.sample_id == "007020")
        assert rec20.has_ct_dicom
        assert rec20.ct_dicom_instance_number == "3"

    def test_asymmetric_ct_pet_coverage(self, tmp_path):
        """CT and PET having different slice_id sets must NOT shift each
        other's DICOM index."""
        import pydicom
        from pydicom.dataset import Dataset

        raw = tmp_path / "Data"
        test_root = tmp_path / "subset_B"
        test_root.mkdir()
        (test_root / "part_CT" / "train_data" / "ImageSet" / "PNG").mkdir(parents=True)
        (test_root / "part_PET" / "ImageSet" / "PNG").mkdir(parents=True)

        # Patient 009: CT has slices {010, 015, 020}, PET has slices {005, 015}
        # If union index is used, PET would get shifted indices.
        for sid in ("009010", "009015", "009020"):
            _touch(raw / "subset_A" / "part_CT" / "train_data" / "ImageSet" / "PNG" / f"{sid}.png")
        for sid in ("009005", "009015"):
            _touch(raw / "subset_A" / "part_PET" / "ImageSet" / "PNG" / f"{sid}.png")

        dcm_root = tmp_path / "dicom_root"

        # CT DICOM: 3 slices (matches 3 CT PNGs)
        pdir_ct = dcm_root / "part_CT" / "train_data" / "ImageSet" / "DICOM" / "009"
        pdir_ct.mkdir(parents=True)
        for i in range(3):
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.9.{i}"
            ds.SeriesInstanceUID = "1.2.826.0.1.3680043.9.7132.9"
            ds.InstanceNumber = str(i + 1)
            ds.ImagePositionPatient = ["0.0", "0.0", str(float(3 - i))]
            ds.PatientID = "009"
            ds.Modality = "CT"
            pydicom.dcmwrite(str(pdir_ct / f"{i + 1}.dcm"), ds, little_endian=True, implicit_vr=True)

        # PET DICOM: 2 slices (matches 2 PET PNGs)
        pdir_pet = dcm_root / "part_PET" / "ImageSet" / "DICOM" / "009"
        pdir_pet.mkdir(parents=True)
        for i in range(2):
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.9.7132.9.{i + 100}"
            ds.SeriesInstanceUID = "1.2.826.0.1.3680043.9.7132.9"
            ds.InstanceNumber = str(i + 1)
            ds.ImagePositionPatient = ["0.0", "0.0", str(float(2 - i))]
            ds.PatientID = "009"
            ds.Modality = "PT"
            pydicom.dcmwrite(str(pdir_pet / f"{i + 1}.dcm"), ds, little_endian=True, implicit_vr=True)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm_root, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )

        # 009005: PET-only sample. slice_id=5 → PET local_index=0 → PET DICOM[0]
        rec005 = next(r for r in records if r.sample_id == "009005")
        assert not rec005.has_ct_png
        assert rec005.has_pet_png
        assert rec005.has_pet_dicom, f"PET DICOM should match: {rec005.mapping_status}"
        assert rec005.pet_dicom_instance_number == "1"
        # CT DICOM should NOT be attached (sample doesn't have CT PNG)
        assert not rec005.has_ct_dicom

        # 009010: CT-only sample. slice_id=10 → CT local_index=0 → CT DICOM[0]
        rec010 = next(r for r in records if r.sample_id == "009010")
        assert rec010.has_ct_png
        assert not rec010.has_pet_png
        assert rec010.has_ct_dicom, f"CT DICOM should match: {rec010.mapping_status}"
        assert rec010.ct_dicom_instance_number == "1"
        assert not rec010.has_pet_dicom

        # 009015: BOTH CT and PET. slice_id=15
        #   CT: {010:0, 015:1, 020:2} → local_index=1 → CT DICOM[1] → InstanceNumber=2
        #   PET: {005:0, 015:1} → local_index=1 → PET DICOM[1] → InstanceNumber=2
        rec015 = next(r for r in records if r.sample_id == "009015")
        assert rec015.has_ct_png and rec015.has_pet_png
        assert rec015.has_ct_dicom, f"CT DICOM: {rec015.mapping_status}"
        assert rec015.has_pet_dicom, f"PET DICOM: {rec015.mapping_status}"
        assert rec015.ct_dicom_instance_number == "2"
        assert rec015.pet_dicom_instance_number == "2"


# ═══ 11. val_ratio edge cases ═══════════════════════════════════════════

class TestValRatio:
    def test_val_ratio_zero_allowed(self, tmp_path):
        """val_ratio=0 should produce 0 val patients."""
        records = [
            SampleRecord(sample_id=f"00{i}001", patient_id=f"00{i}", slice_id=1, split="")
            for i in range(5)
        ]
        assign_splits(records, test_sample_ids=set(), val_ratio=0.0, seed=42)
        val_records = [r for r in records if r.split == "val"]
        assert len(val_records) == 0
        train_records = [r for r in records if r.split == "train"]
        assert len(train_records) == 5


# ═══ 12. integration ════════════════════════════════════════════════════

class TestBuildSampleRecords:
    def test_full_integration(self, tmp_path):
        raw = _build_fake_raw(tmp_path)
        test_root = _build_fake_test_root(tmp_path)
        dcm = _build_fake_dicom_root(tmp_path)

        records = build_sample_records(
            raw_root=raw, dicom_root=dcm, test_png_root=test_root,
            sample_id_regex=DEFAULT_SAMPLE_ID_REGEX,
            dicom_slice_order="z_desc", dicom_index_offset=0,
        )
        # subset_A: 5 samples (004002-004004, 005002-005003)
        # subset_B: 3 samples (001002, 001003, 002002)
        # label: 001002 (overlaps with subset_B), 004002, 004003
        # Total unique ids: 5 + 3 = 8
        sample_ids = {r.sample_id for r in records}
        assert len(sample_ids) == 8
        assert "004002" in sample_ids
        assert "001002" in sample_ids

        # 004002 should have DICOM attached
        rec = next(r for r in records if r.sample_id == "004002")
        assert rec.has_ct_png and rec.has_pet_png
        assert rec.has_ct_dicom, f"DICOM match failed: {rec.mapping_status}"
        assert rec.has_pet_dicom
        assert rec.mapping_status == "ok"
