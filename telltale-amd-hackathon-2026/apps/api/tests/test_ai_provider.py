"""Central AI Analysis Agent - slice S1 tests: provider-neutral boundary + factory.

Covers the approved slice-S1 contract: an internal provider abstraction
(``AIProvider`` protocol, ``ProviderRequest`` / ``ProviderResult`` /
``ProviderUsage`` / ``ProviderError`` with a :class:`ProviderFailureClass` enum and
timeout/cancellation subclasses), a factory that currently supports ONLY the
deterministic ``template`` provider for all five features (explain,
evidence_review, analyze, report, ask), a facts-only request context with no
reachable stores/models/filesystem, and byte-identical behavior: the adapter is
delegated to the unchanged template providers so candidates, prompt digests,
identity, and the five routes' responses are identical to the pre-slice code.

No downloads, no GPUs, no network, no new provider: snapshots are built through the
real assess/v1 service; the shared slice-D builders/helpers are reused.
"""

from __future__ import annotations

import inspect
from dataclasses import fields as dataclass_fields
from datetime import timedelta

import pytest
from pydantic import ValidationError

from telltale_api.core.config import PLANNED_AI_PROVIDERS, SUPPORTED_AI_PROVIDERS, Settings
from telltale_api.schemas.ai import AiAskOutput, AiExplainOutput
from telltale_api.services.ai_analyze_template import TemplateAnalyzeProvider
from telltale_api.services.ai_ask_template import TemplateAskProvider
from telltale_api.services.ai_provider import (
    AIProvider,
    ProviderCancelled,
    ProviderError,
    ProviderFailureClass,
    ProviderFeature,
    ProviderRequest,
    ProviderResult,
    ProviderTimeout,
    ProviderUsage,
    TemplateAIProvider,
    coerce_feature,
    get_ai_provider,
    resolve_provider_name,
)
from telltale_api.services.ai_report_template import TemplateReportProvider
from telltale_api.services.ai_review_template import TemplateEvidenceReviewProvider
from telltale_api.services.ai_template import TemplateExplainProvider, project_assessment

from telltale_api.services.ai_amd_provider import (
    AmdExplainProvider,
    AmdAnalyzeProvider,
    AmdReviewProvider,
)
from tests.test_ai_report import (  # shared slice-D builders/helpers (no side effects)
    AUTH_ALICE,
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

FEATURES = (
    ProviderFeature.EXPLAIN,
    ProviderFeature.EVIDENCE_REVIEW,
    ProviderFeature.ANALYZE,
    ProviderFeature.REPORT,
    ProviderFeature.ASK,
)


# --------------------------------------------------------------------- factory


def test_factory_returns_template_provider_for_every_feature():
    for feature in FEATURES:
        provider = get_ai_provider(feature)
        assert isinstance(provider, TemplateAIProvider)
        assert isinstance(provider, AIProvider)  # satisfies the protocol
        assert provider.feature is feature


def test_factory_accepts_kind_and_path_style_names():
    assert get_ai_provider("explain").feature is ProviderFeature.EXPLAIN
    assert get_ai_provider("evidence_review").feature is ProviderFeature.EVIDENCE_REVIEW
    assert get_ai_provider("evidence-review").feature is ProviderFeature.EVIDENCE_REVIEW
    assert get_ai_provider("analyze").feature is ProviderFeature.ANALYZE
    assert get_ai_provider("report").feature is ProviderFeature.REPORT
    assert get_ai_provider("ask").feature is ProviderFeature.ASK


def test_factory_rejects_unknown_feature():
    with pytest.raises(ValueError):
        get_ai_provider("openai")
    with pytest.raises(ValueError):
        coerce_feature("nope")


def test_provider_identity_is_template_with_no_model():
    for feature in FEATURES:
        provider = get_ai_provider(feature)
        assert provider.model is None


def test_factory_routes_without_importing_transport():
    # The routing module must stay free of any transport/client dependency: the AMD
    # client lives in its own lazily-imported module (slice S3).
    from telltale_api.services import ai_provider as module

    source = inspect.getsource(module)
    assert "import httpx" not in source
    assert "import requests" not in source
    assert "from ..core.config import" in source


# --------------------------------------------------------------------- request


def test_request_carries_facts_only_projected_context():
    snapshot = make_snapshot(("sensor", "acoustic"))
    provider = get_ai_provider("explain")
    request = provider.build_request((snapshot,))
    assert isinstance(request, ProviderRequest)
    assert request.feature is ProviderFeature.EXPLAIN
    assert request.context == (snapshot,)
    # The ONLY data a provider reads is the facts-only projection of the snapshot.
    assert request.facts == (project_assessment(snapshot),)


def test_request_exposes_no_store_model_or_filesystem_handles():
    names = {f.name for f in dataclass_fields(ProviderRequest)}
    assert names == {
        "feature", "context", "question", "scope", "lookback_days",
        "window_start", "window_end", "history_truncated", "machine_id",
    }
    forbidden = {"store", "db", "conn", "registry", "models", "model", "path", "paths",
                 "settings", "secret", "token", "audio", "fleet", "client", "session"}
    assert names.isdisjoint(forbidden)


def test_provider_module_imports_no_infrastructure():
    from telltale_api.services import ai_provider as module

    source = inspect.getsource(module)
    for forbidden in (
        "ai_output_store", "assessment_store", "ModelRegistry", "import torch",
        "onnxruntime", "import httpx", "import requests", "urllib.request",
        "from .fleet", "open(",
    ):
        assert forbidden not in source, forbidden
    # S2: the factory reads the provider NAME from configuration - settings, not
    # infrastructure - so importing core.config is allowed and expected.
    assert "from ..core.config import" in source


def test_provider_instances_have_no_infrastructure_attributes():
    provider = get_ai_provider("ask")
    for attr in ("store", "db", "conn", "registry", "settings", "paths"):
        assert not hasattr(provider, attr), attr


# --------------------------------------------------------------------- result


def test_result_carries_candidate_and_usage_metadata():
    snapshot = make_snapshot(("sensor",))
    provider = get_ai_provider("explain")
    request = provider.build_request((snapshot,))
    result = provider.generate(request)
    assert isinstance(result, ProviderResult)
    assert isinstance(result.candidate, AiExplainOutput)
    assert result.provider == "template" and result.model is None
    assert isinstance(result.usage, ProviderUsage)
    assert result.usage.latency_ms is not None and result.usage.latency_ms >= 0.0
    # The deterministic template provider reports no token accounting.
    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


def test_generate_rejects_a_mismatched_feature():
    provider = get_ai_provider("explain")
    wrong = get_ai_provider("ask").build_request((make_snapshot(("sensor",)),))
    with pytest.raises(ProviderError) as excinfo:
        provider.generate(wrong)
    assert excinfo.value.failure_class is ProviderFailureClass.PROVIDER_ERROR


# --------------------------------------------------------------------- failures


def test_failure_class_enum_members():
    assert {member.value for member in ProviderFailureClass} == {
        "timeout", "network", "cancelled", "auth", "rate_limited",
        "provider_error", "malformed", "provider_refusal",
    }
    # str-mixin keeps them comparable with plain strings (routing/logging friendly).
    assert ProviderFailureClass.TIMEOUT == "timeout"


def test_failure_class_retry_policy():
    assert ProviderError(ProviderFailureClass.TIMEOUT, "x").retryable is True
    assert ProviderError(ProviderFailureClass.NETWORK, "x").retryable is True
    assert ProviderError(ProviderFailureClass.RATE_LIMITED, "x").retryable is True
    assert ProviderError(ProviderFailureClass.AUTH, "x").retryable is False
    assert ProviderError(ProviderFailureClass.MALFORMED, "x").retryable is False
    assert ProviderError(ProviderFailureClass.PROVIDER_REFUSAL, "x").retryable is False
    assert ProviderError(ProviderFailureClass.PROVIDER_ERROR, "x").retryable is False
    # explicit override wins
    assert ProviderError(ProviderFailureClass.AUTH, "x", retryable=True).retryable is True


def test_timeout_and_cancellation_types():
    timeout = ProviderTimeout()
    assert isinstance(timeout, ProviderError)
    assert timeout.failure_class is ProviderFailureClass.TIMEOUT
    assert timeout.retryable is True

    cancelled = ProviderCancelled()
    assert isinstance(cancelled, ProviderError)
    assert cancelled.failure_class is ProviderFailureClass.CANCELLED
    assert cancelled.retryable is False


# ------------------------------------------------------------- byte-identity


def test_explain_review_analyze_outputs_are_byte_identical():
    snapshot = make_snapshot(("sensor", "acoustic", "bearing"))
    cases = (
        ("explain", TemplateExplainProvider()),
        ("evidence_review", TemplateEvidenceReviewProvider()),
        ("analyze", TemplateAnalyzeProvider()),
    )
    direct = {
        "explain": TemplateExplainProvider().explain,
        "evidence_review": TemplateEvidenceReviewProvider().review,
        "analyze": TemplateAnalyzeProvider().analyze,
    }
    for kind, impl in cases:
        provider = get_ai_provider(kind)
        request = provider.build_request((snapshot,))
        result = provider.generate(request)
        assert result.candidate.model_dump() == direct[kind](snapshot).model_dump(), kind
        assert provider.prompt_sha256(request) == impl.prompt_sha256(snapshot), kind


def test_report_output_and_digest_are_byte_identical():
    snapshot = make_snapshot(("sensor", "acoustic"))
    window_end = snapshot.created_at
    window_start = window_end - timedelta(days=30)
    impl = TemplateReportProvider()
    provider = get_ai_provider("report")
    request = provider.build_request(
        (snapshot,), machine_id=MACHINE, window_start=window_start,
        window_end=window_end, history_truncated=False, lookback_days=30,
    )
    result = provider.generate(request)
    expected = impl.report(
        [snapshot], window_start=window_start, window_end=window_end,
        history_truncated=False, lookback_days=30,
    )
    assert result.candidate.model_dump() == expected.model_dump()
    assert provider.prompt_sha256(request) == impl.prompt_sha256(
        MACHINE, [snapshot], window_start=window_start, window_end=window_end,
        history_truncated=False, lookback_days=30,
    )


def test_ask_output_and_digest_are_byte_identical():
    snapshot = make_snapshot(("sensor", "acoustic"))
    window_end = snapshot.created_at
    window_start = window_end - timedelta(days=30)
    question = "Is the machine healthy?"
    impl = TemplateAskProvider()
    provider = get_ai_provider("ask")
    request = provider.build_request(
        (snapshot,), scope="machine", question=question, lookback_days=30,
        window_start=window_start, window_end=window_end,
        history_truncated=False, machine_id=MACHINE,
    )
    result = provider.generate(request)
    assert isinstance(result.candidate, AiAskOutput)
    expected = impl.ask(
        [snapshot], scope="machine", question=question, lookback_days=30,
        window_start=window_start, window_end=window_end,
        history_truncated=False, machine_id=MACHINE,
    )
    assert result.candidate.model_dump() == expected.model_dump()
    assert provider.prompt_sha256(request) == impl.prompt_sha256(
        [snapshot], scope="machine", question=question, lookback_days=30, machine_id=MACHINE,
    )


# --------------------------------------------------------- route regressions


def test_all_five_routes_respond_with_template_identity(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        aid = str(snapshot.assessment_id)
        calls = (
            (URL_EXPLAIN, {"assessment_id": aid}, "explain", "explain/v1"),
            (URL_REVIEW, {"assessment_id": aid}, "evidence_review", "evidence-review/v1"),
            (URL_ANALYZE, {"assessment_id": aid}, "analyze", "analyze/v1"),
            (URL_REPORT, {"machine_id": MACHINE}, "report", "report/v1"),
            (URL_ASK, {"assessment_id": aid, "question": "Is the machine healthy?"}, "ask", "ask/v1"),
        )
        for url, body, feature, prompt_version in calls:
            resp = client.post(url, json=body, headers=AUTH_ALICE)
            assert resp.status_code == 201, (url, resp.text)
            payload = resp.json()
            assert payload["feature"] == feature, url
            assert payload["provider"] == "template", url
            assert payload["model"] is None, url
            assert payload["prompt_version"] == prompt_version, url
            assert resp.headers["Location"] == f"/api/v1/ai/outputs/{payload['ai_output_id']}"


# ------------------------------------------------------- configuration (slice S2)


def _settings(**over):
    # _env_file=None keeps these tests independent of any local .env file.
    return Settings(_env_file=None, **over)


def test_supported_and_planned_provider_sets():
    assert SUPPORTED_AI_PROVIDERS == frozenset({"template", "amd"})
    assert PLANNED_AI_PROVIDERS == frozenset({"openai"})


def test_default_provider_is_template(monkeypatch):
    monkeypatch.delenv("TELLTALE_AI_PROVIDER", raising=False)
    assert _settings().ai_provider == "template"


def test_explicit_template_selection_works(monkeypatch):
    assert _settings(ai_provider="template").ai_provider == "template"
    # path/name normalisation: case and surrounding whitespace are tolerated
    assert _settings(ai_provider="  TeMpLaTe ").ai_provider == "template"
    monkeypatch.setenv("TELLTALE_AI_PROVIDER", "template")
    assert Settings(_env_file=None).ai_provider == "template"


@pytest.mark.parametrize("value", ["openai"])
def test_unavailable_provider_fails_closed_at_configuration(value):
    with pytest.raises(ValidationError) as excinfo:
        _settings(ai_provider=value)
    # clear message, and it echoes only the provider NAME (never a secret)
    assert value in str(excinfo.value)
    assert "not supported" in str(excinfo.value)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"amd_api_key": None},
        {"amd_model": None},
        {"amd_api_key": None, "amd_model": None},
        {"amd_api_key": "   "},
        {"amd_model": "   "},
    ],
)
def test_amd_without_key_or_model_fails_closed(over):
    with pytest.raises(ValidationError):
        _settings(ai_provider="amd", **over)


def test_amd_with_key_and_model_is_accepted():
    settings = _settings(ai_provider="amd", amd_api_key="rc-" + "a" * 48, amd_model="some-amd-model")
    assert settings.ai_provider == "amd"
    assert settings.amd_model == "some-amd-model"
    # the secret is never serialized in clear text
    assert "rc-" + "a" * 48 not in settings.model_dump_json()


@pytest.mark.parametrize(
    "value", ["http://developer.amd.com.cn/radeon/api/v1", "https://evil.example/radeon/api/v1",
              "https://developer.amd.com.cn/radeon/api/v1/extra", "ftp://developer.amd.com.cn/x"],
)
def test_amd_base_url_must_be_the_documented_https_endpoint(value):
    with pytest.raises(ValidationError):
        _settings(ai_provider="amd", amd_api_key="rc-" + "a" * 48, amd_model="m", amd_api_base_url=value)


@pytest.mark.parametrize(
    "field,value", [("ai_timeout_s", 0), ("ai_timeout_s", 121), ("ai_max_output_tokens", 0),
                    ("ai_max_output_tokens", 4001), ("ai_max_retries", 4), ("ai_retry_max_s", 31)],
)
def test_ai_transport_limits_fail_closed_above_the_maxima(field, value):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


@pytest.mark.parametrize("value", ["", "   ", "garbage", "templat", "amdx", "openai2"])
def test_invalid_provider_value_fails_closed_at_configuration(value):
    with pytest.raises(ValidationError):
        _settings(ai_provider=value)


def test_env_configured_unavailable_provider_fails_closed(monkeypatch):
    monkeypatch.setenv("TELLTALE_AI_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_factory_uses_configured_template_for_all_features():
    settings = _settings(ai_provider="template")
    for feature in FEATURES:
        provider = get_ai_provider(feature, settings=settings)
        assert isinstance(provider, TemplateAIProvider)
        assert provider.feature is feature


def test_resolve_provider_name_precedence():
    settings = _settings(ai_provider="template")
    assert resolve_provider_name(settings=settings) == "template"
    assert resolve_provider_name("  TEMPLATE ", settings=settings) == "template"


@pytest.mark.parametrize("value", ["openai", "garbage", ""])
def test_factory_fails_closed_for_unavailable_provider(value):
    with pytest.raises(ProviderError) as excinfo:
        get_ai_provider("explain", provider=value)
    assert excinfo.value.failure_class is ProviderFailureClass.PROVIDER_ERROR
    assert "template" in excinfo.value.message  # tells the operator what is supported


def test_factory_keeps_report_and_ask_on_template_under_amd():
    # AMD is approved for explain, analyze, and evidence-review (per AGENTS.md 6.13).
    # Report and Ask always keep the deterministic template provider.
    settings = _settings(ai_provider="amd", amd_api_key="rc-" + "a" * 48, amd_model="some-amd-model")
    # Explain and analyze should use AMD provider
    provider_explain = get_ai_provider(ProviderFeature.EXPLAIN, settings=settings)
    provider_analyze = get_ai_provider(ProviderFeature.ANALYZE, settings=settings)
    provider_evidence_review = get_ai_provider(ProviderFeature.EVIDENCE_REVIEW, settings=settings)
    # Report and Ask should stay on template
    provider_report = get_ai_provider(ProviderFeature.REPORT, settings=settings)
    provider_ask = get_ai_provider(ProviderFeature.ASK, settings=settings)
    assert isinstance(provider_explain, AmdExplainProvider)
    assert isinstance(provider_analyze, AmdAnalyzeProvider)
    assert isinstance(provider_evidence_review, AmdReviewProvider)
    assert isinstance(provider_report, TemplateAIProvider)
    assert isinstance(provider_ask, TemplateAIProvider)


def test_factory_fails_closed_for_amd_explain_without_usable_credentials():
    # Explicit amd override with template settings (no key/model) must fail closed.
    with pytest.raises(ProviderError) as excinfo:
        get_ai_provider("explain", provider="amd", settings=_settings(ai_provider="template"))
    assert excinfo.value.failure_class is ProviderFailureClass.PROVIDER_ERROR


def test_legacy_oauth_placeholders_are_ignored(monkeypatch):
    secret = "should-never-be-echoed-3f9c"
    monkeypatch.setenv("TELLTALE_AMD_CLIENT_SECRET", secret)
    monkeypatch.setenv("TELLTALE_AMD_TOKEN_URL", "https://example.invalid/token")
    monkeypatch.setenv("TELLTALE_AMD_CLIENT_ID", "client")
    monkeypatch.setenv("TELLTALE_AMD_SCOPES", "scope")
    settings = Settings(_env_file=None)
    for attr in ("amd_client_secret", "amd_token_url", "amd_client_id", "amd_scopes"):
        assert not hasattr(settings, attr), attr
    assert secret not in settings.model_dump_json()


def test_amd_secret_is_redacted_everywhere():
    secret = "rc-" + "a" * 48
    settings = _settings(ai_provider="amd", amd_api_key=secret, amd_model="some-amd-model")
    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in settings.model_dump_json()
    assert "**********" in settings.model_dump_json()


def test_factory_error_message_never_contains_secret_values(monkeypatch):
    secret = "rc-" + "b" * 48
    monkeypatch.setenv("TELLTALE_AMD_API_KEY", secret)
    monkeypatch.setenv("TELLTALE_AMD_MODEL", "some-amd-model")
    with pytest.raises(ValidationError) as excinfo:
        _settings(ai_provider="openai")
    assert secret not in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo2:
        _settings(ai_provider="amd", amd_model=None)
    assert secret not in str(excinfo2.value)


def test_routes_work_with_explicit_template_selection(client_factory):
    with client_factory(ai_provider="template") as client:
        snapshot = seed(client, ("sensor",))
        resp = client.post(
            URL_EXPLAIN,
            json={"assessment_id": str(snapshot.assessment_id)},
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        payload = resp.json()
        assert payload["provider"] == "template"
        assert payload["model"] is None
