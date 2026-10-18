"""Foundation tests for immutable assessments: schemas, versioned migrations,
append-only store and the pure assess/v1 quality/status rules.

No endpoints, no inference orchestration, no model training and no downloads:
everything here builds typed outputs in memory and uses temporary SQLite files
under D: (TEMP/TMP are pointed there by the test run).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from telltale_api.schemas.acoustic import (
    AcousticAnalysis,
    BandDeviation,
    FaultDiagnosis,
    FaultProbability,
    Series,
    Spectrogram,
    Waveform,
)
from telltale_api.schemas.assessment import (
    SCHEMA_VERSION,
    Assessment,
    EvidenceCompleteness,
    EvidenceDraft,
    ModelIdentity,
    ModalityDraft,
)
from telltale_api.schemas.sensor import Contribution, FailureModeScore, SensorAssessment
from telltale_api.services.assess import (
    RULE_VERSION,
    bearing_indication,
    create_assessment,
)
from telltale_api.services.assessment_store import (
    MIGRATIONS,
    AssessmentAlreadyExists,
    AssessmentStore,
    run_migrations,
)

NOW = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)
OBS = datetime(2026, 10, 15, 13, 59, 0, tzinfo=timezone.utc)

P_EXACT = 0.9577348106516856
THRESHOLD_EXACT = 0.45691827248299344
SCORE_EXACT = -0.41239871293741261


def sensor_output(risk: str = "low", p: float = 0.002423118420741755) -> SensorAssessment:
    return SensorAssessment(
        failure_probability=p,
        risk_level=risk,  # type: ignore[arg-type]
        decision_threshold=THRESHOLD_EXACT,
        failure_modes=[
            FailureModeScore(code="TWF", name="Tool wear", probability=0.1, threshold=0.5, flagged=False)
        ],
        contributions=[
            Contribution(feature="tool_wear_min", label="Tool wear", value=212.0, contribution=0.51)
        ],
        base_value=-2.0,
        derived={"tool_wear_slope": 1.0},
        warnings=[],
    )


def acoustic_output(status: str = "normal") -> AcousticAnalysis:
    return AcousticAnalysis(
        machine_type="fan",
        score=SCORE_EXACT,
        threshold=1.339198992587626,
        severity=0.6,
        status=status,  # type: ignore[arg-type]
        is_anomalous=status == "anomalous",
        duration_s=10.0,
        sample_rate=16000,
        peak_time_s=1.0,
        frame_errors=Series(times=[0.0, 1.0], values=[0.1, 0.2]),
        spectrogram=Spectrogram(times=[0.0], band_hz=[100.0], values=[[0.5]]),
        band_deviation=BandDeviation(band_hz=[100.0], z=[0.3]),
        waveform=Waveform(duration_s=10.0, min=[-0.9], max=[0.8]),
    )


def bearing_output(
    pred: str = "normal",
    pset: list[str] | None = None,
    uncertain: bool = False,
    probs: list[FaultProbability] | None = None,
) -> FaultDiagnosis:
    if pset is None:
        pset = ["normal"]
    if probs is None:
        probs = [FaultProbability(class_name="normal", label="Normal", probability=0.8)]
    return FaultDiagnosis(
        predicted_class=pred,
        label=pred,
        confidence=0.8,
        prediction_set=list(pset),
        prediction_set_labels=list(pset),
        uncertain=uncertain,
        segment_agreement=0.9,
        segments=5,
        duration_s=4.0,
        probabilities=probs,
    )


def full_identity() -> ModelIdentity:
    return ModelIdentity(
        name="fan-ensemble-v2",
        version="2.3.1",
        trained_at=datetime(2026, 10, 14, 21, 30, tzinfo=timezone.utc),
        artifact_fingerprint="a" * 64,
        preprocessing_fingerprint="b" * 64,
        training_dataset="mimii_due",
    )


def sensor_draft(
    risk: str = "low",
    source: str = "uploaded",
    observed: datetime | None = OBS,
    identity: ModelIdentity | None = None,
) -> ModalityDraft:
    return ModalityDraft(
        kind="sensor",
        availability="available",
        reason_code="ok",
        source_mode=source,  # type: ignore[arg-type]
        observed_at=observed,
        inference_at=NOW,
        model_identity=identity or full_identity(),
        output=sensor_output(risk),
    )


def simple_assessment(
    machine_id: str = "m-1",
    source: str = "uploaded",
    expected: list[str] | None = None,
    modalities: list[ModalityDraft] | None = None,
    evidence: list[EvidenceDraft] | None = None,
    created_at: datetime | None = NOW,
) -> Assessment:
    return create_assessment(
        machine_id=machine_id,
        source_mode=source,  # type: ignore[arg-type]
        expected_modalities=expected or ["sensor"],  # type: ignore[arg-type]
        modalities=modalities if modalities is not None else [sensor_draft(source=source)],
        evidence=evidence if evidence is not None else [
            EvidenceDraft(
                kind="failure_probability",
                value=P_EXACT,
                unit="probability",
                source="sensor",
                source_mode=source,  # type: ignore[arg-type]
                observed_at=OBS,
            )
        ],
        created_at=created_at,
    )


def flag_codes(a: Assessment) -> set[str]:
    return {f.code for f in a.quality_flags}


# ---------------------------------------------------------------- migrations


def test_migration_fresh_db(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    store = AssessmentStore(db)
    conn = sqlite3.connect(db)
    versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    names = [row[0] for row in conn.execute("SELECT name FROM schema_migrations ORDER BY version")]
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert versions == [v for v, _, _ in MIGRATIONS]
    assert names == [n for _, n, _ in MIGRATIONS]
    assert {"assessments", "schema_migrations"} <= tables
    store.close()


def test_migration_preserves_pre_existing_alerts_db(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE alerts (id TEXT PRIMARY KEY, created_at TEXT, json TEXT)")
    conn.execute(
        "INSERT INTO alerts (id, created_at, json) VALUES (?, ?, ?)",
        ("alert-1", "2026-10-15T10:00:00+00:00", '{"status": "open"}'),
    )
    conn.commit()
    conn.close()

    store = AssessmentStore(db)

    conn = sqlite3.connect(db)
    alert_rows = conn.execute("SELECT id, json FROM alerts").fetchall()
    versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    conn.close()
    assert alert_rows == [("alert-1", '{"status": "open"}')]
    assert versions == [v for v, _, _ in MIGRATIONS]
    store.close()


def test_migration_idempotent_and_version_recorded(tmp_path: Path) -> None:
    db = tmp_path / "idem.db"
    conn = sqlite3.connect(db, isolation_level=None)
    first = run_migrations(conn)
    second = run_migrations(conn)
    third = run_migrations(conn)
    assert first == [v for v, _, _ in MIGRATIONS]
    assert second == []
    assert third == []
    rows = conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    conn.close()
    assert len(rows) == len(MIGRATIONS)
    for (version, name, applied_at), (v, n, _) in zip(rows, MIGRATIONS):
        assert (version, name) == (v, n)
        parsed = datetime.fromisoformat(applied_at)
        assert parsed.tzinfo is not None  # recorded with a timezone-aware UTC stamp


# ---------------------------------------------------------------- store


def test_store_insert_get_roundtrip_and_stamps(tmp_path: Path) -> None:
    a = simple_assessment()
    store = AssessmentStore(tmp_path / "store.db")
    store.insert(a)
    got = store.get(a.assessment_id)
    assert got is not None
    assert got.model_dump_json() == a.model_dump_json()
    assert got.schema_version == SCHEMA_VERSION == "1.0"
    assert got.rule_version == RULE_VERSION == "assess/v1"
    # Full UUIDs, not shortened ids.
    assert len(str(got.assessment_id)) == 36
    assert len(str(got.modalities[0].result_id)) == 36
    assert len(str(got.evidence[0].evidence_id)) == 36
    store.close()


def test_store_list_pagination_bounded_stable_ordering(tmp_path: Path) -> None:
    store = AssessmentStore(tmp_path / "page.db")
    group_a = [
        simple_assessment(machine_id="m-a", created_at=NOW + timedelta(seconds=1)) for _ in range(5)
    ]
    group_b = [simple_assessment(machine_id="m-b", created_at=NOW) for _ in range(200)]
    for a in group_a + group_b:
        store.insert(a)

    def key(a: Assessment) -> tuple[datetime, str]:
        return (a.created_at, str(a.assessment_id))

    expected = [str(a.assessment_id) for a in sorted(group_a + group_b, key=key, reverse=True)]

    # Bounded: a huge limit is clamped to MAX_PAGE (200), not honoured verbatim.
    capped = store.list(limit=10_000)
    assert len(capped) == 200
    assert [str(a.assessment_id) for a in capped] == expected[:200]

    # Paging: disjoint pages that together cover everything exactly once.
    pages = [store.list(limit=60, offset=o) for o in (0, 60, 120, 180)]
    flat = [str(a.assessment_id) for page in pages for a in page]
    assert flat == expected
    assert len(set(flat)) == 205

    # Stable order: repeated calls return the same sequence.
    assert [str(a.assessment_id) for a in store.list(limit=200)] == expected[:200]

    # Negative offsets are clamped, not honoured.
    assert [str(a.assessment_id) for a in store.list(limit=5, offset=-10)] == expected[:5]

    # machine_id filter.
    only_a = store.list(machine_id="m-a", limit=50)
    assert {a.machine_id for a in only_a} == {"m-a"}
    assert len(only_a) == 5
    store.close()


def test_store_unknown_id_and_duplicate_id(tmp_path: Path) -> None:
    store = AssessmentStore(tmp_path / "dup.db")
    assert store.get(uuid4()) is None

    a = simple_assessment()
    store.insert(a)
    with pytest.raises(AssessmentAlreadyExists) as exc_info:
        store.insert(a)
    assert exc_info.value.assessment_id == a.assessment_id

    # Same id via an independent (deep) copy is still a typed conflict.
    with pytest.raises(AssessmentAlreadyExists):
        store.insert(a.model_copy(deep=True))
    store.close()


def test_store_append_only_surface() -> None:
    for name in ("update", "replace", "delete", "upsert", "remove", "set"):
        assert not hasattr(AssessmentStore, name), f"store must stay append-only ({name})"


def test_snapshot_isolation_mutation_cannot_touch_record(tmp_path: Path) -> None:
    output = sensor_output(p=P_EXACT)
    draft = sensor_draft()
    draft.output = output
    evidence = EvidenceDraft(
        kind="failure_probability",
        value=P_EXACT,
        unit="probability",
        source="sensor",
        source_mode="uploaded",
        observed_at=OBS,
    )
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[draft],
        evidence=[evidence],
        created_at=NOW,
    )
    stored_before = a.model_dump_json()
    store = AssessmentStore(tmp_path / "iso.db")
    store.insert(a)

    # Mutate the ORIGINAL input objects after the insert.
    output.failure_probability = 0.99
    output.derived["sneaky"] = 5.0
    evidence.value = 0.5
    draft.source_mode = "dataset_demo"  # type: ignore[assignment]

    # Mutate the Assessment object that was inserted.
    a.overall_status = "critical"  # type: ignore[assignment]
    a.modalities[0].output.failure_probability = 0.11  # type: ignore[attr-defined]

    got = store.get(a.assessment_id)
    assert got is not None
    assert got.modalities[0].output.failure_probability == P_EXACT  # type: ignore[attr-defined]
    assert "sneaky" not in got.modalities[0].output.derived  # type: ignore[attr-defined]
    assert got.evidence[0].value == P_EXACT
    assert got.overall_status == "normal"
    assert got.source_mode == "uploaded"
    assert got.model_dump_json() == stored_before

    # Mutating a RETURNED snapshot cannot alter the stored record either.
    got.overall_status = "critical"  # type: ignore[assignment]
    again = store.get(a.assessment_id)
    assert again is not None and again.overall_status == "normal"
    store.close()


def test_new_result_cannot_mutate_older_snapshot(tmp_path: Path) -> None:
    older = simple_assessment(created_at=NOW)
    newer = simple_assessment(created_at=NOW + timedelta(seconds=1), modalities=[sensor_draft(risk="high")])
    store = AssessmentStore(tmp_path / "older.db")
    store.insert(older)
    store.insert(newer)
    older_json = store.get(older.assessment_id).model_dump_json()  # type: ignore[union-attr]

    # Deep-mutate the newer result everywhere.
    newer.overall_status = "unknown"  # type: ignore[assignment]
    newer.modalities[0].output.failure_probability = 0.777  # type: ignore[attr-defined]
    newer.evidence[0].value = 0.1  # type: ignore[index]

    reloaded = store.get(older.assessment_id)
    assert reloaded is not None
    assert reloaded.model_dump_json() == older_json
    assert reloaded.overall_status == "normal"
    assert reloaded.modalities[0].output.failure_probability != 0.777  # type: ignore[attr-defined]
    store.close()


def test_restart_persistence_identical_content(tmp_path: Path) -> None:
    db = tmp_path / "restart.db"
    a = simple_assessment()
    first = AssessmentStore(db)
    first.insert(a)
    first.close()

    second = AssessmentStore(db)  # "restart": fresh connection, migrations re-run (no-op)
    stored = second.get(a.assessment_id)
    assert stored is not None
    assert stored.model_dump_json() == a.model_dump_json()
    assert [x.assessment_id for x in second.list()] == [a.assessment_id]
    second.close()


# ------------------------------------------------------- availability / status


def test_availability_completeness_and_missing_not_normal() -> None:
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor", "acoustic"],
        modalities=[sensor_draft()],
        created_at=NOW,
    )
    assert a.overall_status == "unknown"  # missing required evidence is never "normal"
    assert a.evidence_completeness == EvidenceCompleteness(expected=2, present=1, ratio=0.5)
    assert flag_codes(a) == {"EVIDENCE_INCOMPLETE"}
    assert "acoustic:unavailable:not_submitted" in a.status_reasons
    assert "incomplete_evidence_precludes_normal" in a.status_reasons
    # The individual normal indication is preserved even though overall is unknown.
    assert a.modalities[0].output.risk_level == "low"  # type: ignore[attr-defined]
    assert a.modalities[1].availability == "unavailable"
    assert a.modalities[1].reason_code == "not_submitted"

    # No available evidence at all -> unknown + explicit flag + explicit reason.
    b = create_assessment(
        machine_id="m-2",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[
            ModalityDraft(
                kind="sensor",
                availability="unavailable",
                reason_code="model_not_loaded",
                source_mode="uploaded",
            )
        ],
        created_at=NOW,
    )
    assert b.overall_status == "unknown"
    assert "NO_AVAILABLE_MODALITY" in flag_codes(b)
    assert "EVIDENCE_INCOMPLETE" in flag_codes(b)
    assert "no_usable_evidence" in b.status_reasons
    assert b.evidence_completeness.ratio == 0.0
    assert "sensor:unavailable:model_not_loaded" in b.status_reasons


def test_incomplete_flag_is_warning_and_critical_indication_is_preserved() -> None:
    # Incomplete/missing evidence raises a *warning* flag; "critical" stays reserved
    # for actual model-indicated criticality (here: sensor risk=high).
    a = create_assessment(
        machine_id="m-3",
        source_mode="simulated",
        expected_modalities=["sensor", "acoustic", "bearing"],
        modalities=[sensor_draft(risk="high", source="simulated")],
        created_at=NOW,
    )
    assert a.overall_status == "critical"
    codes = flag_codes(a)
    assert "EVIDENCE_INCOMPLETE" in codes
    assert "SOURCE_ILLUSTRATIVE" in codes
    flag = next(f for f in a.quality_flags if f.code == "EVIDENCE_INCOMPLETE")
    assert flag.severity == "warning"
    assert a.evidence_completeness.present == 1
    assert a.evidence_completeness.expected == 3
    assert "acoustic:unavailable:not_submitted" in a.status_reasons
    assert "bearing:unavailable:not_submitted" in a.status_reasons

    watch = create_assessment(
        machine_id="m-4",
        source_mode="uploaded",
        expected_modalities=["sensor", "acoustic"],
        modalities=[sensor_draft(risk="medium")],
        created_at=NOW,
    )
    assert watch.overall_status == "watch"
    assert "EVIDENCE_INCOMPLETE" in flag_codes(watch)


def test_availability_reason_code_matrix() -> None:
    def build(**kw):
        base = dict(
            kind="sensor",
            source_mode="uploaded",
            observed_at=OBS,
            inference_at=NOW,
        )
        base.update(kw)
        return ModalityDraft(**base)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):  # omitted modality may not be "ok"
        build(availability="available", reason_code="not_submitted", output=sensor_output())
    with pytest.raises(ValidationError):  # unloaded model is not an inference error
        build(availability="unavailable", reason_code="inference_error")
    with pytest.raises(ValidationError):
        build(availability="invalid", reason_code="inference_error")
    with pytest.raises(ValidationError):
        build(availability="error", reason_code="invalid_input")
    with pytest.raises(ValidationError):  # available requires an output
        build(availability="available", reason_code="ok")
    with pytest.raises(ValidationError):  # non-available must not carry an output
        build(availability="unavailable", reason_code="not_submitted", output=sensor_output())
    with pytest.raises(ValidationError):  # available must not carry error_detail
        build(availability="available", reason_code="ok", output=sensor_output(), error_detail="boom")
    # The supported states themselves are valid.
    assert build(availability="unsupported", reason_code="unsupported_input").availability == "unsupported"
    assert build(availability="error", reason_code="inference_error", error_detail="x").availability == "error"


# ---------------------------------------------------------------- timestamps


def test_timestamp_validation_and_normalization() -> None:
    naive = datetime(2026, 10, 15, 12, 0, 0)
    with pytest.raises(ValidationError):  # naive modality timestamp rejected
        sensor_draft(observed=naive)
    with pytest.raises(ValidationError):  # naive evidence timestamp rejected
        EvidenceDraft(
            kind="k", value=1.0, source="s", source_mode="uploaded", observed_at=naive
        )
    with pytest.raises(ValidationError):  # malformed string rejected
        sensor_draft(observed="not-a-timestamp")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):  # naive created_at rejected
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft()],
            created_at=naive,
        )

    # Aware timestamps are normalized to UTC.
    shifted = ModalityDraft(
        kind="sensor",
        availability="unavailable",
        reason_code="not_submitted",
        source_mode="uploaded",
        observed_at=datetime(2026, 10, 15, 18, 30, tzinfo=timezone(timedelta(hours=5))),
    )
    assert shifted.observed_at == datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc)
    assert shifted.observed_at.utcoffset() == timedelta(0)  # type: ignore[union-attr]

    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft()],
        created_at=datetime(2026, 10, 15, 16, 0, tzinfo=timezone(timedelta(hours=5))),
    )
    assert a.created_at == datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc)

    # Default created_at: server-generated UTC "now".
    b = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft()],
    )
    assert b.created_at.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - b.created_at).total_seconds()) < 120

    # Missing timestamp stays None and raises a quality flag (never fabricated).
    c = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft(observed=None)],
        evidence=[
            EvidenceDraft(kind="k", value=1.0, source="s", source_mode="uploaded", observed_at=OBS)
        ],
        created_at=NOW,
    )
    assert c.modalities[0].observed_at is None
    assert "TIMESTAMP_MISSING" in flag_codes(c)
    flag = next(f for f in c.quality_flags if f.code == "TIMESTAMP_MISSING")
    assert flag.severity == "warning"

    d = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft()],
        evidence=[EvidenceDraft(kind="k", value=1.0, source="s", source_mode="uploaded")],
        created_at=NOW,
    )
    assert d.evidence[0].observed_at is None
    assert "TIMESTAMP_MISSING" in flag_codes(d)

    # No invented freshness/future-skew policy: extremes are accepted, not rejected.
    far_future = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft(observed=datetime(2027, 6, 1, tzinfo=timezone.utc))],
        created_at=NOW,
    )
    assert far_future.modalities[0].observed_at == datetime(2027, 6, 1, tzinfo=timezone.utc)
    far_past = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft(observed=datetime(1999, 1, 1, tzinfo=timezone.utc))],
        created_at=NOW,
    )
    assert far_past.modalities[0].observed_at == datetime(1999, 1, 1, tzinfo=timezone.utc)


def test_timestamp_text_ordering_fixed_precision_microseconds(tmp_path: Path) -> None:
    # Same second, different microseconds (incl. exact-zero fraction).
    stamps = [
        datetime(2026, 10, 15, 14, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 15, 14, 0, 0, 500000, tzinfo=timezone.utc),
        datetime(2026, 10, 15, 14, 0, 0, 999999, tzinfo=timezone.utc),
        datetime(2026, 10, 15, 14, 0, 0, 1, tzinfo=timezone.utc),
    ]
    db = tmp_path / "micro.db"
    store = AssessmentStore(db)
    for i, stamp in enumerate(stamps):
        store.insert(simple_assessment(machine_id=f"m-{i}", created_at=stamp))
    listed = store.list(limit=10)
    assert [a.created_at for a in listed] == sorted(stamps, reverse=True)

    # Stored text uses ONE exact format (always 6-digit fraction, always +00:00),
    # so lexicographic order == chronological order without assuming mixed
    # precision sorts correctly.
    conn = sqlite3.connect(db)
    texts = [row[0] for row in conn.execute("SELECT created_at FROM assessments ORDER BY created_at")]
    conn.close()
    assert len(texts) == 4
    for text in texts:
        assert text.endswith("+00:00")
        fraction = text.split(".")[1].split("+")[0]
        assert len(fraction) == 6  # fixed microsecond precision on every row
    assert texts == sorted(texts)  # lexicographic == chronological
    store.close()


def test_wrong_typed_output_rejected() -> None:
    # An available modality requires its ORIGINAL typed output - never another
    # modality's output type.
    with pytest.raises(ValidationError):
        ModalityDraft(
            kind="sensor",
            availability="available",
            reason_code="ok",
            source_mode="uploaded",
            output=acoustic_output(),  # wrong type for a sensor modality
        )
    with pytest.raises(ValidationError):
        ModalityDraft(
            kind="bearing",
            availability="available",
            reason_code="ok",
            source_mode="uploaded",
            output=sensor_output(),  # wrong type for a bearing modality
        )


def test_expected_and_modality_duplicates_cannot_hide_missing_evidence() -> None:
    # Duplicate expected kinds are rejected: completeness must stay deterministic.
    with pytest.raises(ValueError):
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor", "sensor"],
            modalities=[sensor_draft()],
            created_at=NOW,
        )
    # Duplicate records for one kind are rejected: one slot cannot overwrite another.
    with pytest.raises(ValueError):
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft(), sensor_draft()],
            created_at=NOW,
        )
    # A record for a modality outside the expected list is rejected.
    with pytest.raises(ValueError):
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft(), bearing_draft()],
            created_at=NOW,
        )
    # Expected-but-omitted modalities are forced into explicit unavailable records:
    # missing evidence cannot be hidden by shrinking the record list.
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor", "acoustic", "bearing"],
        modalities=[sensor_draft()],
        created_at=NOW,
    )
    assert {m.kind for m in a.modalities} == {"sensor", "acoustic", "bearing"}
    assert a.evidence_completeness.expected == 3
    assert a.evidence_completeness.present == 1
    assert a.overall_status == "unknown"


# --------------------------------------------------------- provenance / quality


def test_source_provenance_and_illustrative_flags() -> None:
    # source_mode is provenance: it stays "uploaded" although the model was
    # trained on MIMII/demo data. training_dataset is a separate field.
    a = simple_assessment(source="uploaded")
    assert a.source_mode == "uploaded"
    assert a.modalities[0].source_mode == "uploaded"
    assert a.modalities[0].model_identity is not None
    assert a.modalities[0].model_identity.training_dataset == "mimii_due"
    assert a.evidence[0].source_mode == "uploaded"
    assert a.evidence[0].illustrative is False
    assert "SOURCE_ILLUSTRATIVE" not in flag_codes(a)
    assert "MIXED_SOURCE_PROVENANCE" not in flag_codes(a)
    assert a.quality_flags == []  # clean, complete, fully-identified upload

    # dataset_demo/simulated inputs are labelled illustrative (server-computed).
    b = create_assessment(
        machine_id="m-2",
        source_mode="dataset_demo",
        expected_modalities=["sensor"],
        modalities=[sensor_draft(source="dataset_demo")],
        evidence=[
            EvidenceDraft(kind="failure_probability", value=0.1, source="sensor", source_mode="dataset_demo")
        ],
        created_at=NOW,
    )
    assert "SOURCE_ILLUSTRATIVE" in flag_codes(b)
    assert b.modalities[0].illustrative is True
    assert b.evidence[0].illustrative is True

    # Unrelated-source combined evidence is a DISTINCT warning, not the illustrative flag.
    c = create_assessment(
        machine_id="m-3",
        source_mode="simulated",
        expected_modalities=["sensor", "acoustic"],
        modalities=[
            sensor_draft(source="simulated"),
            ModalityDraft(
                kind="acoustic",
                availability="available",
                reason_code="ok",
                source_mode="uploaded",
                observed_at=OBS,
                inference_at=NOW,
                model_identity=full_identity(),
                output=acoustic_output("watch"),
            ),
        ],
        created_at=NOW,
    )
    codes = flag_codes(c)
    assert "MIXED_SOURCE_PROVENANCE" in codes
    assert "SOURCE_ILLUSTRATIVE" in codes
    mixed = next(f for f in c.quality_flags if f.code == "MIXED_SOURCE_PROVENANCE")
    illustrative = next(f for f in c.quality_flags if f.code == "SOURCE_ILLUSTRATIVE")
    assert mixed.severity == "warning"
    assert illustrative.severity == "info"
    assert c.overall_status == "watch"  # combined status still computed


def test_original_numbers_preserved_through_storage(tmp_path: Path) -> None:
    output = acoustic_output("anomalous")
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["acoustic"],
        modalities=[
            ModalityDraft(
                kind="acoustic",
                availability="available",
                reason_code="ok",
                source_mode="uploaded",
                observed_at=OBS,
                inference_at=NOW,
                model_identity=full_identity(),
                output=output,
            )
        ],
        evidence=[
            EvidenceDraft(
                kind="sound_score", value=SCORE_EXACT, unit="z", source="acoustic", source_mode="uploaded"
            )
        ],
        created_at=NOW,
    )
    assert output.score == SCORE_EXACT
    assert output.threshold == 1.339198992587626

    store = AssessmentStore(tmp_path / "numbers.db")
    store.insert(a)
    got = store.get(a.assessment_id)
    assert got is not None
    got_output = got.modalities[0].output
    assert isinstance(got_output, AcousticAnalysis)
    assert got_output.score == SCORE_EXACT  # exact float round-trip, no re-scaling
    assert got_output.threshold == 1.339198992587626
    assert got.evidence[0].value == SCORE_EXACT

    sensor = simple_assessment()
    store.insert(sensor)
    got_sensor = store.get(sensor.assessment_id)
    assert got_sensor is not None
    out = got_sensor.modalities[0].output
    assert isinstance(out, SensorAssessment)
    assert out.failure_probability == 0.002423118420741755
    assert out.decision_threshold == THRESHOLD_EXACT
    assert out.contributions[0].contribution == 0.51
    assert out.contributions[0].value == 212.0
    store.close()


def test_missing_model_identity_is_flagged_not_invented() -> None:
    full = simple_assessment()
    assert "MODEL_IDENTITY_INCOMPLETE" not in flag_codes(full)

    version_only = ModalityDraft(
        kind="sensor",
        availability="available",
        reason_code="ok",
        source_mode="uploaded",
        observed_at=OBS,
        model_identity=ModelIdentity(version="0.1.0"),  # version alone is NOT sufficient
        output=sensor_output(),
    )
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[version_only],
        created_at=NOW,
    )
    assert "MODEL_IDENTITY_INCOMPLETE" in flag_codes(a)
    # The missing parts are represented explicitly as None, never invented.
    identity = a.modalities[0].model_identity
    assert identity is not None
    assert identity.artifact_fingerprint is None
    assert identity.preprocessing_fingerprint is None

    no_identity_draft = ModalityDraft(
        kind="sensor",
        availability="available",
        reason_code="ok",
        source_mode="uploaded",
        observed_at=OBS,
        inference_at=NOW,
        output=sensor_output(),
    )
    no_identity = create_assessment(
        machine_id="m-2",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[no_identity_draft],
        created_at=NOW,
    )
    assert "MODEL_IDENTITY_INCOMPLETE" in flag_codes(no_identity)
    assert no_identity.modalities[0].model_identity is None  # explicit, not fabricated


def test_model_limitations_preserved_as_evidence_limitations() -> None:
    draft = sensor_draft()
    draft.limitations = ["Trained on synthetic AI4I data; real-factory generalization unverified"]
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[draft],
        created_at=NOW,
    )
    assert a.modalities[0].limitations == [
        "Trained on synthetic AI4I data; real-factory generalization unverified"
    ]
    assert "MODEL_LIMITATIONS_PRESENT" in flag_codes(a)
    limitation = next(f for f in a.quality_flags if f.code == "MODEL_LIMITATIONS_PRESENT")
    assert limitation.severity == "info"
    # The rules only surface caveats; they never invent accuracy numbers.
    assert "accuracy" not in limitation.explanation.lower() or "no accuracy" in limitation.explanation.lower()


# ------------------------------------------------------------ bearing policy


BEARING_CASES = [
    ("normal-only", dict(pred="normal", pset=["normal"]), "normal", []),
    (
        "fault-set-fault-pred",
        dict(pred="inner_race", pset=["inner_race", "normal"]),
        "watch",
        ["bearing:fault_classes_in_set"],
    ),
    (
        "ambiguous-normal-pred",
        dict(pred="normal", pset=["normal", "inner_race"]),
        "unknown",
        ["bearing:ambiguous_normal_with_faults"],
    ),
    ("empty-set", dict(pset=[]), "unknown", ["bearing:empty_prediction_set"]),
    (
        "pred-not-in-set",
        dict(pred="outer_race", pset=["inner_race"]),
        "unknown",
        ["bearing:inconsistent_output"],
    ),
    (
        "classifier-uncertain",
        dict(pred="normal", pset=["normal"], uncertain=True),
        "unknown",
        ["bearing:classifier_uncertain"],
    ),
    (
        "no-class-probabilities",
        dict(pred="inner_race", pset=["inner_race"], probs=[]),
        "unknown",
        ["bearing:no_class_probabilities"],
    ),
]


@pytest.mark.parametrize("name,kwargs,expected_indication,expected_reasons", BEARING_CASES)
def test_conservative_bearing_mapping(
    name: str, kwargs: dict, expected_indication: str, expected_reasons: list[str]
) -> None:
    output = bearing_output(**kwargs)
    indication, reasons = bearing_indication(output)
    assert indication == expected_indication, name
    assert reasons == expected_reasons, name
    assert indication != "critical", name

    # As the only modality the overall status follows the conservative mapping
    # and can never be "critical".
    a = create_assessment(
        machine_id="m-bear",
        source_mode="uploaded",
        expected_modalities=["bearing"],
        modalities=[
            ModalityDraft(
                kind="bearing",
                availability="available",
                reason_code="ok",
                source_mode="uploaded",
                observed_at=OBS,
                inference_at=NOW,
                model_identity=full_identity(),
                output=output,
            )
        ],
        created_at=NOW,
    )
    assert a.overall_status == expected_indication, name
    assert a.overall_status != "critical", name
    if expected_indication == "unknown":
        assert "BEARING_UNCERTAIN" in flag_codes(a), name
        assert "uncertain_evidence_precludes_normal" in a.status_reasons, name
    if expected_reasons:
        assert set(expected_reasons) <= set(a.status_reasons), name


def test_bearing_fault_set_alone_is_not_confirmed_fault() -> None:
    # A conformal set containing fault classes (without a fault prediction being
    # confirmed) stays "watch": review priority, never critical, never silent "normal".
    output = bearing_output(pred="inner_race", pset=["inner_race"])
    indication, reasons = bearing_indication(output)
    assert indication == "watch"
    assert reasons == ["bearing:fault_classes_in_set"]


# ---------------------------------------------------------- server-owned rules


def test_server_owned_fields_cannot_be_supplied() -> None:
    with pytest.raises(ValidationError):  # evidence ids are server-generated
        EvidenceDraft(kind="k", value=1.0, source="s", source_mode="uploaded", evidence_id=uuid4())
    with pytest.raises(ValidationError):  # illustrative is server-computed
        EvidenceDraft(kind="k", value=1.0, source="s", source_mode="uploaded", illustrative=True)
    with pytest.raises(ValidationError):  # result ids are server-generated
        ModalityDraft(
            kind="sensor",
            availability="available",
            reason_code="ok",
            source_mode="uploaded",
            output=sensor_output(),
            result_id=uuid4(),
        )
    with pytest.raises(ValidationError):  # invented score outside the typed output
        ModalityDraft(
            kind="sensor",
            availability="available",
            reason_code="ok",
            source_mode="uploaded",
            output=sensor_output(),
            failure_probability=0.99,
        )
    with pytest.raises(ValidationError):  # versions are not client-suppliable
        ModalityDraft(
            kind="sensor",
            availability="available",
            reason_code="ok",
            source_mode="uploaded",
            output=sensor_output(),
            schema_version="9.9",
        )
    with pytest.raises(TypeError):  # overall status is computed, never accepted
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft()],
            overall_status="normal",
        )
    with pytest.raises(TypeError):  # reserved field stays reserved
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft()],
            approved_playbook_item_ids=["pb-1"],
        )
    with pytest.raises(ValueError):  # empty expected list has no deterministic completeness
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=[],
        )
    with pytest.raises(ValidationError):  # machine_id is required and non-empty
        create_assessment(
            machine_id="",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft()],
        )
    with pytest.raises(ValidationError):  # a modality outside the expected list is rejected
        create_assessment(
            machine_id="m-1",
            source_mode="uploaded",
            expected_modalities=["sensor"],
            modalities=[sensor_draft(), bearing_draft()],
        )

    a = simple_assessment()
    b = simple_assessment()
    assert a.assessment_id != b.assessment_id  # fresh full UUID per assessment
    assert a.modalities[0].result_id != b.modalities[0].result_id
    assert a.evidence[0].evidence_id != b.evidence[0].evidence_id
    assert a.approved_playbook_item_ids == []  # reserved: no playbook mapping yet
    assert "confidence" not in Assessment.model_fields
    assert "confidence" not in EvidenceCompleteness.model_fields  # completeness != confidence


def bearing_draft() -> ModalityDraft:
    return ModalityDraft(
        kind="bearing",
        availability="available",
        reason_code="ok",
        source_mode="uploaded",
        observed_at=OBS,
        inference_at=NOW,
        model_identity=full_identity(),
        output=bearing_output(),
    )


def test_no_llm_dependence() -> None:
    import telltale_api.services.assess as assess_mod
    import telltale_api.services.assessment_store as store_mod

    for mod in (assess_mod, store_mod):
        assert "httpx" not in mod.__dict__
        assert "narrator" not in mod.__dict__
        assert "Settings" not in mod.__dict__

    here = Path(__file__).resolve().parents[1] / "telltale_api" / "services"
    for filename in ("assess.py", "assessment_store.py"):
        text = (here / filename).read_text(encoding="utf-8")
        for forbidden in ("httpx", "narrator", "openai", "TELLTALE_LLM"):
            assert forbidden not in text, f"{filename} must not reference {forbidden}"

    # The pure rules run with no app, settings or LLM in scope.
    a = create_assessment(
        machine_id="m-1",
        source_mode="uploaded",
        expected_modalities=["sensor"],
        modalities=[sensor_draft()],
        created_at=NOW,
    )
    assert a.overall_status == "normal"
    assert a.rule_version == "assess/v1"
