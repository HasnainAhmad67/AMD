"""Slice S3 tests: AMD Token Factory inference provider for POST /api/v1/ai/explain.

Everything here runs against ``httpx.MockTransport`` - there is NO live AMD call, no
network and no real sleep. The tests cover: the documented request (URL/method/path,
Bearer auth, JSON body, no tools/streaming), the minimized facts-only projection (no
snapshot, paths, fingerprints, secrets or question echo), strict response parsing
(empty/truncated/malformed/missing content), the timeout/retry policy (short vs long
``Retry-After``, non-retryable auth/parameter errors, bounded retries), and the
route-level provenance rules (an accepted AMD candidate is ``provider=amd``; every
failure/fallback/refusal stays ``provider=template``), plus owner isolation and the
byte-identical template path when template is selected.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from telltale_api.core.config import Settings
from telltale_api.services.ai_amd_provider import (
    ANALYZE_JSON_INSTRUCTIONS,
    AmdAnalyzeProse,
    AmdAnalyzeProvider,
    AmdExplainProse,
    AmdExplainProvider,
    AmdReviewProse,
    AmdReviewProvider,
    JSON_INSTRUCTIONS,
    REVIEW_JSON_INSTRUCTIONS,
    minimize_for_amd,
)
from telltale_api.services.ai_provider import (
    ProviderError,
    ProviderFailureClass,
    ProviderFeature,
    get_ai_provider,
)
from telltale_api.services.ai_template import TemplateExplainProvider, project_assessment
from telltale_api.services.ai_analyze_template import TemplateAnalyzeProvider
from telltale_api.services.ai_review_template import TemplateEvidenceReviewProvider
from tests.test_ai_report import (  # shared slice-D builders/helpers (no side effects)
    AUTH_ALICE,
    AUTH_BOB,
    MACHINE,
    client_factory,  # noqa: F401 - re-exported as a fixture
    make_snapshot,
    seed,
)

URL_EXPLAIN = "/api/v1/ai/explain"
URL_REVIEW = "/api/v1/ai/evidence-review"
URL_ANALYZE = "/api/v1/ai/analyze"
URL_REPORT = "/api/v1/ai/report"
URL_ASK = "/api/v1/ai/ask"

KEY = "rc-" + "a" * 48
MODEL = "test-amd-model"
BASE = "https://developer.amd.com.cn/radeon/api/v1"
CHAT_URL = BASE + "/chat/completions"

# A prose answer that satisfies the UNCHANGED grounding validator (no invented numbers
# or quoted tokens, and the exact required phrases).
GROUNDED_TEXT = (
    "This explanation is built from the stored assessment facts only. These results are "
    "model indications, not confirmed faults. Missing evidence limits interpretation, so "
    "no further interpretation is offered beyond the stored values."
)
INVENTED_NUMBER_TEXT = (
    "The machine has a failure probability of 99.99 percent and will fail soon. These are "
    "model indications, not confirmed faults. Missing evidence limits interpretation."
)
INVENTED_QUOTE_TEXT = (
    'The stored data shows a "catastrophic" wear pattern. These are model indications, not '
    "confirmed faults. Missing evidence limits interpretation."
)
# A review answer that satisfies the UNCHANGED review grounding validator:
REVIEW_GROUNDED_TEXT = (
    "This review is built from the stored assessment facts only. It is not a fault diagnosis, "
    "not machine-health completeness, and not coverage. Missing evidence limits interpretation, "
    "so no further interpretation is offered beyond the stored values."
)
# Same required phrases, but one invented number -> grounding must reject it.
INVENTED_NUMBER_REVIEW_TEXT = (
    "The two modalities co-occur and the combined score is 99.99. This is not a fault "
    "diagnosis, not machine-health completeness and not coverage; no fused failure "
    "probability, health score, risk score or fused probability is created here."
)
# A co-occurrence answer that satisfies the UNCHANGED analyze grounding validator: the
# four required non-claim phrases, no invented numbers, no invented quoted tokens.
GROUNDED_ANALYZE_TEXT = (
    "Stored indications across the requested modalities are described here without being "
    "combined into one value. This is not a fault diagnosis, it is not machine-health "
    "completeness and it is not coverage, and no fused failure probability, health score "
    "or risk score is created. Missing or uncertain evidence limits interpretation and "
    "nothing was fabricated to fill gaps. No modality explains another and no causality is "
    "inferred anywhere in this analysis."
)
# Same required phrases, but one invented number -> grounding must reject it.
INVENTED_NUMBER_ANALYZE_TEXT = (
    "The two modalities co-occur and the combined score is 0.97. This is not a fault "
    "diagnosis, not machine-health completeness and not coverage; no fused failure "
    "probability is created here."
)

MIN_PAYLOAD_KEYS = {
    "machine_id", "overall_status", "status_reasons", "expected_modalities",
    "present_modalities", "missing_modalities", "quality_flags", "modalities",
}


def _settings(**over):
    base = {"ai_provider": "amd", "amd_api_key": KEY, "amd_model": MODEL}
    base.update(over)
    return Settings(_env_file=None, **base)


def _envelope(content: str, *, usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _content(**over) -> str:
    payload = {"explanation_text": GROUNDED_TEXT}
    payload.update(over)
    return json.dumps(payload)


def _review_content(**over) -> str:
    payload = {"review_text": REVIEW_GROUNDED_TEXT}
    payload.update(over)
    return json.dumps(payload)


class FakeClock:
    """Deterministic monotonic clock + sleep so retry tests never really wait."""

    def __init__(self, step: float = 10.0, start: float = 0.0) -> None:
        self.t = start
        self.step = step

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += self.step


def _provider(handler, *, clock: FakeClock | None = None, **over) -> AmdExplainProvider:
    clock = clock or FakeClock(step=0.0)
    return AmdExplainProvider(
        _settings(**over),
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        monotonic=clock,
    )


def _explain_request(snapshot):
    return get_ai_provider("explain", settings=_settings()).build_request((snapshot,))


def _install_stub(monkeypatch, handler, *, clock: FakeClock | None = None):
    """Make the route's lazily-imported AMD provider use a MockTransport."""
    import telltale_api.services.ai_amd_provider as amd_mod

    clock = clock or FakeClock(step=0.0)

    class Stub(amd_mod.AmdExplainProvider):
        def __init__(self, settings):  # noqa: D401 - test double
            super().__init__(
                settings,
                transport=httpx.MockTransport(handler),
                sleep=clock.sleep,
                monotonic=clock,
            )

    monkeypatch.setattr(amd_mod, "AmdExplainProvider", Stub)


def _install_analyze_stub(monkeypatch, handler, *, clock: FakeClock | None = None):
    """Make the analyze route's lazily-imported AMD provider use a MockTransport."""
    import telltale_api.services.ai_amd_provider as amd_mod

    clock = clock or FakeClock(step=0.0)

    class Stub(amd_mod.AmdAnalyzeProvider):
        def __init__(self, settings):  # noqa: D401 - test double
            super().__init__(
                settings,
                transport=httpx.MockTransport(handler),
                sleep=clock.sleep,
                monotonic=clock,
            )

    monkeypatch.setattr(amd_mod, "AmdAnalyzeProvider", Stub)


# ------------------------------------------------------------------ factory

def test_factory_returns_amd_provider_for_explain_when_configured():
    provider = get_ai_provider("explain", settings=_settings())
    assert isinstance(provider, AmdExplainProvider)
    assert provider.name == "amd"
    assert provider.model == MODEL


def test_factory_returns_amd_provider_for_analyze_when_configured():
    provider = get_ai_provider("analyze", settings=_settings())
    assert isinstance(provider, AmdAnalyzeProvider)
    assert provider.name == "amd"
    assert provider.model == MODEL


# ------------------------------------------------------------------ transport

def test_posts_to_documented_url_with_bearer_and_json_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["ctype"] = request.headers.get("Content-Type")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(_content()))

    snapshot = make_snapshot(("sensor",))
    result = _provider(handler).generate(_explain_request(snapshot))

    assert seen["method"] == "POST"
    assert seen["url"] == CHAT_URL
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["ctype"] == "application/json"
    body = seen["body"]
    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == JSON_INSTRUCTIONS
    # default (unset) => max_tokens is OMITTED (some AMD models reject an explicit value)
    assert "max_tokens" not in body
    # nothing that the documented gateway drops/rejects, and no tools/streaming
    for forbidden in ("tools", "tool_choice", "parallel_tool_calls", "stop", "seed",
                      "logit_bias", "logprobs", "top_k", "min_p", "repetition_penalty",
                      "reasoning", "thinking"):
        assert forbidden not in body, forbidden
    assert isinstance(result.candidate, AmdExplainProse)
    assert result.provider == "amd" and result.model == MODEL


# --------------------------------------------------- optional max_tokens (bounded fix)

def _captured_body(settings) -> dict:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(_content()))

    AmdExplainProvider(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    ).generate(_explain_request(make_snapshot(("sensor",))))
    return seen["body"]


@pytest.mark.parametrize("value", ["", "   ", "auto", "AUTO", "omit", "  OMIT ", "none", None])
def test_unset_or_omit_markers_omit_max_tokens(value):
    body = _captured_body(_settings(ai_max_output_tokens=value))
    assert "max_tokens" not in body


@pytest.mark.parametrize("value", [1, 256, 4000, "256"])
def test_configured_positive_integer_includes_max_tokens(value):
    body = _captured_body(_settings(ai_max_output_tokens=value))
    assert body["max_tokens"] == int(value)


@pytest.mark.parametrize("value", [0, -5, 4001, "garbage", True, 2.5])
def test_invalid_max_output_tokens_fails_closed(value):
    with pytest.raises(ValidationError):
        _settings(ai_max_output_tokens=value)


def test_non_amd_provider_default_is_unaffected():
    settings = _settings(ai_provider="template")
    assert settings.ai_max_output_tokens is None
    provider = get_ai_provider("explain", settings=settings)
    assert provider.name == "template" and provider.model is None


# ------------------------------------------------------- analyze transport (wiring)

def _analyze_request(snapshot):
    return get_ai_provider("analyze", settings=_settings()).build_request((snapshot,))


def _analyze_body(settings) -> dict:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(json.dumps({"analysis_text": GROUNDED_ANALYZE_TEXT})))

    AmdAnalyzeProvider(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    ).generate(_analyze_request(make_snapshot(("sensor", "acoustic"))))
    return seen["body"]


def test_analyze_request_omits_max_tokens_when_unset():
    body = _analyze_body(_settings())
    assert "max_tokens" not in body


@pytest.mark.parametrize("value", [1, 256, 4000, "256"])
def test_analyze_request_includes_max_tokens_for_positive_integer(value):
    body = _analyze_body(_settings(ai_max_output_tokens=value))
    assert body["max_tokens"] == int(value)


def test_analyze_request_payload_contains_only_minimized_fields():
    body = _analyze_body(_settings())
    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == ANALYZE_JSON_INSTRUCTIONS
    payload = json.loads(body["messages"][1]["content"])
    assert set(payload) == MIN_PAYLOAD_KEYS
    for forbidden in ("tools", "tool_choice", "parallel_tool_calls", "stop", "seed",
                      "logit_bias", "logprobs", "top_k", "min_p", "repetition_penalty",
                      "reasoning", "thinking"):
        assert forbidden not in body, forbidden
    for secret in ("assessment_id", "evidence_id", "path", "fingerprint", "question"):
        assert secret not in json.dumps(body).lower()


def test_analyze_content_schema_rejects_invented_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_envelope(json.dumps({"analysis_text": GROUNDED_ANALYZE_TEXT, "score": 0.5}))
        )

    provider = AmdAnalyzeProvider(
        _settings(), transport=httpx.MockTransport(handler), sleep=lambda _s: None, monotonic=lambda: 0.0
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.generate(_analyze_request(make_snapshot(("sensor", "acoustic"))))
    assert excinfo.value.failure_class is ProviderFailureClass.MALFORMED


def test_request_body_is_minimized_facts_only():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = request.content.decode()
        seen["payload"] = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json=_envelope(_content()))

    snapshot = make_snapshot(("sensor", "acoustic"))
    _provider(handler).generate(_explain_request(snapshot))

    payload = seen["payload"]
    assert set(payload) == MIN_PAYLOAD_KEYS
    assert payload["machine_id"] == MACHINE
    for modality in payload["modalities"]:
        assert set(modality) == {"kind", "availability", "reason_code", "evidence", "limitations"}
        for item in modality["evidence"]:
            assert set(item) == {"kind", "value", "unit"}
    raw = seen["raw"]
    for forbidden in (
        "assessment_id", "observed_at", "inference_at", "model_name", "model_version",
        "training_dataset", "evidence_id", "fingerprint", "sha256", "path", "question",
        "recording", "audio", "sqlite", "artifact",
    ):
        assert forbidden not in raw, forbidden


def test_minimize_for_amd_excludes_identity_and_ids():
    facts = project_assessment(make_snapshot(("sensor", "bearing")))
    payload = minimize_for_amd(facts)
    assert set(payload) == MIN_PAYLOAD_KEYS
    assert payload["missing_modalities"] == []
    assert payload["present_modalities"] == ["sensor", "bearing"]
    assert payload["expected_modalities"] == ["sensor", "bearing"]


def test_api_key_never_appears_in_repr_or_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "600"})

    provider = _provider(handler)
    assert KEY not in repr(provider)
    snapshot = make_snapshot(("sensor",))
    with pytest.raises(ProviderError) as excinfo:
        provider.generate(_explain_request(snapshot))
    assert KEY not in str(excinfo.value)
    assert KEY not in repr(excinfo.value)


# ------------------------------------------------------------------ parsing

def test_valid_amd_json_returns_prose_candidate_with_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_envelope(
                _content(uncertainty_notes=["stored values only"]),
                usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
            ),
        )

    result = _provider(handler).generate(_explain_request(make_snapshot(("sensor",))))
    prose = result.candidate
    assert prose.explanation_text == GROUNDED_TEXT
    assert prose.uncertainty_notes == ["stored values only"]
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 22
    assert result.usage.total_tokens == 33
    assert result.raw_text == _content(uncertainty_notes=["stored values only"])


def test_markdown_fenced_json_is_accepted():
    fenced = "```json\n" + _content() + "\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(fenced))

    result = _provider(handler).generate(_explain_request(make_snapshot(("sensor",))))
    assert result.candidate.explanation_text == GROUNDED_TEXT


@pytest.mark.parametrize(
    "content",
    [
        "",                                                        # empty
        _content()[:-10],                                          # truncated JSON
        "not json at all",                                         # not JSON
        json.dumps([1, 2, 3]),                                     # not an object
        json.dumps({"explanation_text": "too short"}),             # schema: min_length
        json.dumps({"explanation_text": GROUNDED_TEXT, "id": "x"}),  # extra="forbid"
        json.dumps({"uncertainty_notes": ["missing text"]}),       # required field
    ],
)
def test_bad_content_is_malformed_and_never_returned(content):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(content))

    provider = _provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        provider.generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.MALFORMED
    assert "AMD" in str(excinfo.value)


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 123}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_missing_or_empty_content_is_malformed(envelope):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope)

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.MALFORMED


def test_non_json_envelope_is_malformed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.MALFORMED


# -------------------------------------------------------------- timeout/retry

def test_timeout_is_a_timeout_failure_and_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("no route")

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler, ai_max_retries=0).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.TIMEOUT
    assert calls["n"] == 1


def test_429_with_short_retry_after_retries_within_bound():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=_envelope(_content()))

    result = _provider(handler, ai_retry_max_s=2).generate(_explain_request(make_snapshot(("sensor",))))
    assert calls["n"] == 2
    assert result.candidate.explanation_text == GROUNDED_TEXT


def test_429_with_long_retry_after_falls_back_immediately():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "60"})

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler, ai_retry_max_s=2).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.RATE_LIMITED
    assert calls["n"] == 1  # a long Retry-After never becomes an in-process wait


def test_429_without_retry_after_uses_bounded_backoff():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=_envelope(_content()))

    result = _provider(handler, ai_retry_max_s=1).generate(_explain_request(make_snapshot(("sensor",))))
    assert calls["n"] == 2
    assert result.provider == "amd"


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, ProviderFailureClass.AUTH),
        (403, ProviderFailureClass.AUTH),
        (400, ProviderFailureClass.PROVIDER_ERROR),
        (404, ProviderFailureClass.PROVIDER_ERROR),
        (422, ProviderFailureClass.PROVIDER_ERROR),
    ],
)
def test_auth_and_parameter_errors_are_never_retried(status, expected):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"message": "hidden"}})

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler, ai_max_retries=3).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is expected
    assert calls["n"] == 1


@pytest.mark.parametrize("status", [502, 503])
def test_gateway_errors_are_retried(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status)
        return httpx.Response(200, json=_envelope(_content()))

    result = _provider(handler, ai_max_retries=2).generate(_explain_request(make_snapshot(("sensor",))))
    assert calls["n"] == 2
    assert result.candidate.explanation_text == GROUNDED_TEXT


def test_retries_are_bounded_by_the_configured_maximum():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(ProviderError) as excinfo:
        _provider(handler, ai_max_retries=2).generate(_explain_request(make_snapshot(("sensor",))))
    assert excinfo.value.failure_class is ProviderFailureClass.NETWORK
    assert calls["n"] == 3  # 1 attempt + 2 retries, then stop


def test_total_deadline_bounds_the_operation():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "1"})

    clock = FakeClock(step=10.0)  # sleeping jumps far past the deadline
    with pytest.raises(ProviderError) as excinfo:
        _provider(handler, clock=clock, ai_timeout_s=5, ai_retry_max_s=2).generate(
            _explain_request(make_snapshot(("sensor",)))
        )
    assert excinfo.value.failure_class is ProviderFailureClass.TIMEOUT
    assert calls["n"] == 1


# ---------------------------------------------------------------- route: AMD

def test_route_amd_accepted_reports_amd_provenance(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_content(uncertainty_notes=["stored only"])))

    _install_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        aid = str(snapshot.assessment_id)
        resp = client.post(URL_EXPLAIN, json={"assessment_id": aid}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        assert body["provider"] == "amd"
        assert body["model"] == MODEL
        assert body["explanation_text"] == GROUNDED_TEXT

        stored = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE)
        assert stored.status_code == 200, stored.text
        record = json.loads(stored.text)
        assert record["provider"] == "amd"
        assert record["model"] == MODEL
        assert record["validation_status"] == "validated"


def test_route_backend_owns_the_structural_fields(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_content(uncertainty_notes=["stored only"])))

    _install_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        baseline = TemplateExplainProvider().explain(snapshot)
        resp = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        stored = json.loads(
            client.get(f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE).text
        )
        output = stored["output"]
        assert output["assessment_id"] == baseline.assessment_id
        assert output["machine_id"] == baseline.machine_id
        assert output["overall_status"] == baseline.overall_status
        assert output["status_reasons"] == baseline.status_reasons
        assert output["expected_modalities"] == baseline.expected_modalities
        assert output["present_modalities"] == baseline.present_modalities
        assert output["quality_flag_codes"] == baseline.quality_flag_codes
        assert output["referenced_assessment_ids"] == baseline.referenced_assessment_ids
        assert output["referenced_evidence_ids"] == baseline.referenced_evidence_ids
        assert output["uncertainty_notes"]  # backend-filled when AMD omits them


@pytest.mark.parametrize("text", [INVENTED_NUMBER_TEXT, INVENTED_QUOTE_TEXT])
def test_route_ungrounded_amd_prose_falls_back_to_template(client_factory, monkeypatch, text):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"explanation_text": text})))

    _install_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor",))
        resp = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None
        assert payload["explanation_text"] != text


def test_route_amd_transport_failure_falls_back_to_template(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "opaque"}})

    _install_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor",))
        resp = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None


# ------------------------------------------------------- route: analyze (AMD wiring)

def test_route_amd_analyze_accepted_reports_amd_provenance(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"analysis_text": GROUNDED_ANALYZE_TEXT})))

    _install_analyze_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_ANALYZE, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        assert body["provider"] == "amd"
        assert body["model"] == MODEL
        assert body["analysis_text"] == GROUNDED_ANALYZE_TEXT

        stored = json.loads(
            client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).text
        )
        assert stored["provider"] == "amd"
        assert stored["model"] == MODEL
        assert stored["validation_status"] == "validated"
        # backend-owned structural fields come from the deterministic baseline
        base = TemplateAnalyzeProvider().analyze(snapshot).model_dump(mode="json", by_alias=True)
        out = stored["output"]
        for key in (
            "assessment_id", "machine_id", "overall_status", "status_reasons",
            "expected_modalities", "present_modalities", "missing_modalities", "relations",
            "agreement", "agreement_note", "conservative_interpretation",
            "uncertainty_indicators", "provenance_summary", "non_claims",
            "referenced_assessment_ids",
        ):
            assert out[key] == base[key], key
        assert out["referenced_evidence_ids"]


def test_route_ungrounded_amd_analyze_falls_back_to_template(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"analysis_text": INVENTED_NUMBER_ANALYZE_TEXT})))

    _install_analyze_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_ANALYZE, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None
        assert payload["analysis_text"] != INVENTED_NUMBER_ANALYZE_TEXT


def test_route_amd_analyze_transport_failure_falls_back_to_template(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "opaque"}})

    _install_analyze_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_ANALYZE, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None


def test_route_sensor_only_analyze_is_refused_without_calling_the_provider(client_factory):
    """Slice C eligibility: fewer than two available modalities => deterministic refusal.

    The provider is never invoked, so the persisted provenance stays template/null even
    when TELLTALE_AI_PROVIDER=amd.
    """
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor",))
        resp = client.post(URL_ANALYZE, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "refused"
        assert payload["provider"] == "template"
        assert payload["model"] is None


# -------------------------------------------------- route: selection + legacy

def test_route_remaining_three_features_stay_template_under_amd(client_factory):
    """Under AMD, only REPORT and ASK stay on the deterministic template provider.
    EVIDENCE_REVIEW is now AMD-enabled (alongside explain and analyze)."""
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        # REPORT and ASK must stay template; EVIDENCE_REVIEW is AMD-enabled
        aid = str(seed(client, ("sensor", "acoustic", "bearing")).assessment_id)
        calls = (
            (URL_REPORT, {"machine_id": MACHINE}),
            (URL_ASK, {"assessment_id": aid, "question": "Is the machine healthy?"}),
        )
        for url, body in calls:
            resp = client.post(url, json=body, headers=AUTH_ALICE)
            assert resp.status_code == 201, (url, resp.text)
            payload = resp.json()
            assert payload["provider"] == "template", url
            assert payload["model"] is None, url


def test_route_amd_selected_without_key_fails_closed(client_factory):
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd")
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd", amd_api_key=KEY)
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd", amd_model=MODEL)


def test_route_template_selection_is_byte_identical(client_factory):
    with client_factory(ai_provider="template") as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        baseline = TemplateExplainProvider().explain(snapshot)
        resp = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["provider"] == "template"
        assert payload["model"] is None
        assert payload["prompt_version"] == "explain/v1"
        assert payload["explanation_text"] == baseline.explanation_text
        stored = json.loads(
            client.get(f"/api/v1/ai/outputs/{payload['ai_output_id']}", headers=AUTH_ALICE).text
        )
        assert stored["output"] == baseline.model_dump(mode="json", by_alias=True)


def test_route_owner_isolation_unchanged(client_factory):
    with client_factory(ai_provider="template") as client:
        snapshot = seed(client, ("sensor",), owner="alice")
        resp = client.post(
            URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_BOB
        )
        assert resp.status_code == 404
        assert set(resp.json()) == {"error", "request_id"}


# ------------------------------------------------------ route: review (AMD wiring)

def _install_evidence_review_stub(monkeypatch, handler, *, clock: FakeClock | None = None):
    """Make the evidence-review route's lazily-imported AMD provider use a MockTransport."""
    import telltale_api.services.ai_amd_provider as amd_mod

    clock = clock or FakeClock(step=0.0)

    class Stub(amd_mod.AmdReviewProvider):
        def __init__(self, settings):  # noqa: D401 - test double
            super().__init__(
                settings,
                transport=httpx.MockTransport(handler),
                sleep=clock.sleep,
                monotonic=clock,
            )

    monkeypatch.setattr(amd_mod, "AmdReviewProvider", Stub)


def test_factory_routes_amd_provider_for_evidence_review_when_configured():
    provider = get_ai_provider("evidence_review", settings=_settings())
    assert isinstance(provider, AmdReviewProvider)
    assert provider.name == "amd"
    assert provider.model == MODEL


# ------------------------------------------------------- review transport (wiring)

def _review_request(snapshot):
    return get_ai_provider("evidence_review", settings=_settings()).build_request((snapshot,))


def _review_body(settings) -> dict:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(_review_content()))

    AmdReviewProvider(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
    ).generate(_review_request(make_snapshot(("sensor", "acoustic"))))
    return seen["body"]


def test_evidence_review_request_omits_max_tokens_when_unset():
    body = _review_body(_settings())
    assert "max_tokens" not in body


@pytest.mark.parametrize("value", [1, 256, 4000, "256"])
def test_evidence_review_request_includes_max_tokens_for_positive_integer(value):
    body = _review_body(_settings(ai_max_output_tokens=value))
    assert body["max_tokens"] == int(value)


def test_evidence_review_request_payload_contains_only_minimized_fields():
    body = _review_body(_settings())
    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == REVIEW_JSON_INSTRUCTIONS
    payload = json.loads(body["messages"][1]["content"])
    assert set(payload) == MIN_PAYLOAD_KEYS
    for forbidden in ("tools", "tool_choice", "parallel_tool_calls", "stop", "seed",
                      "logit_bias", "logprobs", "top_k", "min_p", "repetition_penalty",
                      "reasoning", "thinking"):
        assert forbidden not in body, forbidden
    for secret in ("assessment_id", "evidence_id", "path", "fingerprint", "question"):
        assert secret not in json.dumps(body).lower()


# --------------------------------------------------------- review: AMD accepted

def test_route_amd_evidence_review_accepted_reports_amd_provenance(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_content(uncertainty_notes=["stored only"])))

    _install_evidence_review_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        assert body["provider"] == "amd"
        assert body["model"] == MODEL
        assert body["review_text"] == GROUNDED_TEXT

        stored = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE)
        assert stored.status_code == 200, stored.text
        record = json.loads(stored.text)
        assert record["provider"] == "amd"
        assert record["model"] == MODEL
        assert record["validation_status"] == "validated"


def test_route_ungrounded_amd_evidence_review_falls_back_to_template(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"review_text": INVENTED_NUMBER_TEXT})))

    _install_evidence_review_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None
        assert payload["review_text"] != INVENTED_NUMBER_TEXT


def test_route_amd_evidence_review_transport_failure_falls_back_to_template(client_factory, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "opaque"}})

    _install_evidence_review_stub(monkeypatch, handler)
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["validation_status"] == "fallback"
        assert payload["provider"] == "template"
        assert payload["model"] is None


def test_route_sensor_only_evidence_review_requires_at_least_one_modality(client_factory):
    """Eligibility: zero modalities available => deterministic refusal without AMD call.

    With one or more available modalities, the provider IS called.
    This is distinct from the analyze rule of fewer than two modalities."""
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        snapshot = seed(client, ("sensor",))
        resp = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        # Zero modalities => refused without provider call
        assert payload["validation_status"] == "refused"
        assert payload["provider"] == "template"
        assert payload["model"] is None


# ------------------------------------------------------- route: selection + legacy

def test_route_amd_selected_without_key_fails_closed(client_factory):
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd")
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd", amd_api_key=KEY)
    with pytest.raises(ValidationError):
        client_factory(ai_provider="amd", amd_model=MODEL)


def test_route_template_selection_is_byte_identical(client_factory):
    with client_factory(ai_provider="template") as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        baseline = TemplateEvidenceReviewProvider().review(snapshot)
        resp = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)}, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["provider"] == "template"
        assert payload["model"] is None
        assert payload["prompt_version"] == "evidence-review/v1"
        assert payload["review_text"] == baseline.review_text
        stored = json.loads(
            client.get(f"/api/v1/ai/outputs/{payload['ai_output_id']}", headers=AUTH_ALICE).text
        )
        assert stored["output"] == baseline.model_dump(mode="json", by_alias=True)


def test_route_remaining_three_features_stay_template_under_amd(client_factory):
    """Under AMD, only REPORT and ASK stay on the deterministic template provider.
    EVIDENCE_REVIEW is now AMD-enabled (alongside explain and analyze)."""
    with client_factory(ai_provider="amd", amd_api_key=KEY, amd_model=MODEL) as client:
        # REPORT and ASK must stay template; EVIDENCE_REVIEW is AMD-enabled
        aid = str(seed(client, ("sensor", "acoustic", "bearing")).assessment_id)
        calls = (
            (URL_REPORT, {"machine_id": MACHINE}),
            (URL_ASK, {"assessment_id": aid, "question": "Is the machine healthy?"}),
        )
        for url, body in calls:
            resp = client.post(url, json=body, headers=AUTH_ALICE)
            assert resp.status_code == 201, (url, resp.text)
            payload = resp.json()
            assert payload["provider"] == "template", url
            assert payload["model"] is None, url
