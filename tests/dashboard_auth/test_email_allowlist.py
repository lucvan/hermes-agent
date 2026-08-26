"""Tests for the dashboard auth email allowlist.

Covers the four session-minting paths (login, password login, verify,
refresh), the unconfigured no-op case, and the registry lookup boundary —
including that gating does not disturb the registry's identity-conditional
teardown bookkeeping.
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth import email_allowlist as al
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    InvalidCodeError,
    InvalidCredentialsError,
    LoginStart,
    RefreshExpiredError,
    Session,
)

ALLOWED = "lucvan@gmail.com"
STRANGER = "attacker@gmail.com"


def _session(email: str) -> Session:
    return Session(
        user_id="u1",
        email=email,
        display_name="Test",
        org_id="",
        provider="fake",
        expires_at=0,
        access_token="at",
        refresh_token="rt",
    )


class FakeProvider(DashboardAuthProvider):
    name = "fake"
    display_name = "Fake"
    supports_password = True

    def __init__(self, email: str = ALLOWED):
        self.email = email
        self.revoked = False

    def start_login(self, *, redirect_uri: str) -> LoginStart:  # pragma: no cover
        raise NotImplementedError

    def complete_login(self, *, code, state, code_verifier, redirect_uri) -> Session:
        return _session(self.email)

    def complete_password_login(self, *, username: str, password: str) -> Session:
        return _session(self.email)

    def verify_session(self, *, access_token: str):
        return _session(self.email)

    def refresh_session(self, *, refresh_token: str) -> Session:
        return _session(self.email)

    def revoke_session(self, *, refresh_token: str) -> None:
        self.revoked = True


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    """Isolate each test from ambient config and the memoised allowlist."""
    monkeypatch.delenv(al._ENV_VAR, raising=False)
    al.reset_cache()
    yield
    al.reset_cache()


def _configure(monkeypatch, value: str) -> None:
    monkeypatch.setenv(al._ENV_VAR, value)
    al.reset_cache()


# --- unconfigured = inert ------------------------------------------------


def test_unconfigured_returns_provider_unchanged(monkeypatch):
    monkeypatch.setattr(al, "load_allowed_emails", lambda: frozenset())
    p = FakeProvider()
    assert al.wrap(p) is p


# --- the four session-minting paths --------------------------------------


def test_allowed_email_passes_every_path(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    g = al.wrap(FakeProvider(ALLOWED))

    assert g.complete_login(
        code="c", state="s", code_verifier="v", redirect_uri="r"
    ).email == ALLOWED
    assert g.complete_password_login(username="u", password="p").email == ALLOWED
    assert g.verify_session(access_token="at").email == ALLOWED
    assert g.refresh_session(refresh_token="rt").email == ALLOWED


def test_stranger_rejected_on_login(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    g = al.wrap(FakeProvider(STRANGER))
    with pytest.raises(InvalidCodeError):
        g.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")


def test_stranger_rejected_on_password_login(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    g = al.wrap(FakeProvider(STRANGER))
    with pytest.raises(InvalidCredentialsError):
        g.complete_password_login(username="u", password="p")


def test_stranger_rejected_on_verify(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    g = al.wrap(FakeProvider(STRANGER))
    assert g.verify_session(access_token="at") is None


def test_stranger_rejected_on_refresh(monkeypatch):
    """The path that matters most: refresh must not resurrect a session."""
    _configure(monkeypatch, ALLOWED)
    g = al.wrap(FakeProvider(STRANGER))
    with pytest.raises(RefreshExpiredError):
        g.refresh_session(refresh_token="rt")


# --- matching semantics --------------------------------------------------


@pytest.mark.parametrize(
    "configured,claim,expected",
    [
        (ALLOWED, "LucVan@Gmail.com", True),        # case-insensitive
        (f"  {ALLOWED}  ", ALLOWED, True),          # whitespace tolerated
        (f"a@b.com,{ALLOWED}", ALLOWED, True),      # multi-entry
        (ALLOWED, "", False),                       # blank email fails closed
        (ALLOWED, "lucvan@gmail.com.evil.com", False),  # no suffix match
        (ALLOWED, "xlucvan@gmail.com", False),      # no substring match
    ],
)
def test_matching(monkeypatch, configured, claim, expected):
    _configure(monkeypatch, configured)
    g = al.wrap(FakeProvider(claim))
    assert (g.verify_session(access_token="at") is not None) is expected


def test_blank_email_fails_closed_even_with_allowlist():
    assert al.is_allowed("", frozenset({ALLOWED})) is False


def test_empty_allowlist_denies_rather_than_opens():
    """is_allowed is a pure predicate — an empty set must never mean 'any'."""
    assert al.is_allowed(ALLOWED, frozenset()) is False


# --- pass-through --------------------------------------------------------


def test_revoke_delegates(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    inner = FakeProvider(ALLOWED)
    al.wrap(inner).revoke_session(refresh_token="rt")
    assert inner.revoked is True


def test_flags_mirrored(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    inner = FakeProvider(ALLOWED)
    g = al.wrap(inner)
    assert g.name == inner.name
    assert g.display_name == inner.display_name
    assert g.supports_password == inner.supports_password


def test_proxy_is_memoised(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    p = FakeProvider()
    assert al.wrap(p) is al.wrap(p)


def test_session_less_provider_not_wrapped(monkeypatch):
    _configure(monkeypatch, ALLOWED)
    p = FakeProvider()
    p.supports_session = False
    assert al.wrap(p) is p


# --- registry boundary ---------------------------------------------------


def test_registry_lookup_is_gated_and_identity_preserved(monkeypatch):
    """Lookups return a gated proxy, but teardown identity still matches."""
    from hermes_cli.dashboard_auth import registry

    _configure(monkeypatch, ALLOWED)
    inner = FakeProvider(STRANGER)
    inner.name = "fake-registry-test"

    registry.register_global_provider(inner)
    try:
        fetched = registry.get_provider("fake-registry-test")
        # Gated: a stranger's session is filtered out on the way through.
        assert fetched is not inner
        assert fetched.verify_session(access_token="at") is None
        assert any(
            p.name == "fake-registry-test" for p in registry.list_providers()
        )
    finally:
        # The registry stored the *unwrapped* instance, so identity-conditional
        # teardown still works — this is what wrapping at registration broke.
        assert registry.unregister_global_provider("fake-registry-test", inner) is True
