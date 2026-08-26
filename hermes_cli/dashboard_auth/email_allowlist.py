"""Email allowlist enforcement for dashboard auth providers.

The provider protocol deliberately delegates *who may log in* to the
identity provider — ``self_hosted``'s docstring says as much ("The IDP's
own allowlist is authoritative"). That is a reasonable default for an IDP
you run yourself (Authentik, Keycloak, Authelia), where the operator
already controls the user directory.

It is **not** a safe default for a public IDP. Google, for instance, will
happily authenticate any Google account on the internet; the ID token is
perfectly valid and the provider verifies it correctly. Without a check
on the ``email`` claim, "sign in with Google" on a publicly reachable
dashboard means "sign in with *a* Google account".

Restricting that at the IDP is possible (Google's consent screen has a
test-user list) but fragile: it lives in a console one click away from
being published, and for external apps in testing mode Google expires
refresh tokens after 7 days, forcing a weekly re-login. This module moves
the guarantee into Hermes, so it holds regardless of how the IDP is
configured.

Configuration (env wins over config.yaml, matching the rest of the
dashboard auth surface):

    dashboard:
      allowed_emails:
        - you@example.com

    HERMES_DASHBOARD_ALLOWED_EMAILS=you@example.com,ops@example.com

When no allowlist is configured this module is inert and providers are
registered unwrapped — upstream behaviour is unchanged.

Scope: this gates *interactive identities* (``Session``), which is what
an email claim describes. It deliberately does not touch ``verify_token``
/ ``TokenPrincipal``, the non-interactive service-to-service credential
class, which carries no email and is governed by whatever minted it.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    InvalidCodeError,
    InvalidCredentialsError,
    LoginStart,
    RefreshExpiredError,
    Session,
)

logger = logging.getLogger(__name__)

_ENV_VAR = "HERMES_DASHBOARD_ALLOWED_EMAILS"


def _split(raw: Any) -> list[str]:
    """Normalise a list-or-comma-string into lowercased, stripped entries."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = [str(p) for p in raw]
    else:
        return []
    return [p.strip().lower() for p in parts if str(p).strip()]


@functools.lru_cache(maxsize=1)
def load_allowed_emails() -> frozenset[str]:
    """Return the configured allowlist, empty when unconfigured.

    Cached: the allowlist is read once per process, the same lifecycle as
    every other dashboard auth setting (a config change needs a gateway
    restart to take effect). Call :func:`reset_cache` in tests.
    """
    env = _split(os.environ.get(_ENV_VAR, ""))
    if env:
        return frozenset(env)

    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — mirrors the self_hosted provider
        logger.debug(
            "dashboard-auth-allowlist: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return frozenset()

    return frozenset(_split(cfg_get(cfg, "dashboard", "allowed_emails", default=None)))


def reset_cache() -> None:
    """Drop the memoised allowlist. For tests and config reloads.

    Tolerates ``load_allowed_emails`` having been monkeypatched with a
    plain callable (no ``cache_clear``), since this is typically called
    from test teardown where that substitution may still be in place.
    """
    clear = getattr(load_allowed_emails, "cache_clear", None)
    if clear is not None:
        clear()


def is_allowed(email: str, allowed: frozenset[str]) -> bool:
    """Whether ``email`` passes ``allowed``. Fails closed on a blank email.

    An empty allowlist means "unconfigured" and is handled by the caller
    (the provider is left unwrapped), so reaching here with an empty set
    would be a bug — treat it as a denial rather than an open door.
    """
    if not allowed:
        return False
    return (email or "").strip().lower() in allowed


class EmailAllowlistProvider(DashboardAuthProvider):
    """Delegating proxy that filters every ``Session`` an inner provider mints.

    Wrapping at the registry is what makes this airtight: the auth gate
    reaches providers only via ``registry.get_provider`` /
    ``list_providers``, so every path that can produce a session — initial
    login, password login, per-request ``verify_session``, and the
    ``refresh_session`` rotation — passes through here. Enforcing at the
    call sites instead would leave the refresh path as a way to keep a
    session alive after an address stopped being allowed.

    Rejections reuse the protocol's existing failure semantics rather than
    inventing a new one the middleware would not understand:

      * ``complete_login`` → ``InvalidCodeError`` (routes render a failed
        login)
      * ``complete_password_login`` → ``InvalidCredentialsError``
      * ``verify_session`` → ``None`` (gate treats it as no session)
      * ``refresh_session`` → ``RefreshExpiredError`` (gate forces re-login)

    The browser therefore sees a generic login failure, which also avoids
    telling an unknown caller whether an address is on the list. The real
    reason is logged server-side at WARNING.
    """

    def __init__(self, inner: DashboardAuthProvider, allowed: frozenset[str]):
        self._inner = inner
        self.allowed = allowed
        # Instance-level mirrors: the gate reads these off the instance.
        self.name = inner.name
        self.display_name = inner.display_name
        self.supports_password = inner.supports_password
        self.supports_token = inner.supports_token
        self.supports_session = inner.supports_session

    # --- helpers -----------------------------------------------------

    def _check(self, session: Optional[Session], *, path: str) -> bool:
        if session is None:
            return False
        if is_allowed(session.email, self.allowed):
            return True
        logger.warning(
            "dashboard-auth: denied %s for provider %r — email %r is not in "
            "%s (%d entry/entries configured)",
            path,
            self.name,
            session.email or "<empty>",
            _ENV_VAR.lower(),
            len(self.allowed),
        )
        return False

    # --- protocol ----------------------------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        return self._inner.start_login(redirect_uri=redirect_uri)

    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session:
        session = self._inner.complete_login(
            code=code,
            state=state,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )
        if not self._check(session, path="complete_login"):
            raise InvalidCodeError("account is not permitted to access this dashboard")
        return session

    def complete_password_login(self, *, username: str, password: str) -> Session:
        session = self._inner.complete_password_login(
            username=username, password=password
        )
        if not self._check(session, path="complete_password_login"):
            raise InvalidCredentialsError(
                "account is not permitted to access this dashboard"
            )
        return session

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        session = self._inner.verify_session(access_token=access_token)
        return session if self._check(session, path="verify_session") else None

    def refresh_session(self, *, refresh_token: str) -> Session:
        session = self._inner.refresh_session(refresh_token=refresh_token)
        if not self._check(session, path="refresh_session"):
            raise RefreshExpiredError(
                "account is not permitted to access this dashboard"
            )
        return session

    def revoke_session(self, *, refresh_token: str) -> None:
        self._inner.revoke_session(refresh_token=refresh_token)

    def verify_token(self, *, token: str):
        # Non-interactive credential class — no email to gate on. See the
        # module docstring's Scope note.
        return self._inner.verify_token(token=token)


_PROXY_ATTR = "_hermes_email_allowlist_proxy"


def wrap(provider: DashboardAuthProvider) -> DashboardAuthProvider:
    """Return ``provider`` gated by the configured allowlist, or unchanged.

    Session-less providers (``supports_session=False``, e.g. a bearer-token
    credential) mint no interactive identity, so there is nothing to gate.

    Applied at the registry's *lookup* boundary rather than at registration,
    so the stored object stays the caller's own instance and the registry's
    identity-conditional bookkeeping (``unregister_global_provider``,
    ``restore_registration``, which compare with ``is``) keeps working. The
    proxy is memoised on the provider so repeated lookups — this is on the
    per-request path — do not allocate a new one each time.
    """
    allowed = load_allowed_emails()
    if not allowed or not getattr(provider, "supports_session", True):
        return provider

    cached = getattr(provider, _PROXY_ATTR, None)
    if isinstance(cached, EmailAllowlistProvider) and cached.allowed == allowed:
        return cached

    proxy = EmailAllowlistProvider(provider, allowed)
    try:
        setattr(provider, _PROXY_ATTR, proxy)
    except Exception:  # noqa: BLE001 — __slots__ / frozen provider: proceed unmemoised
        pass
    logger.info(
        "dashboard-auth: email allowlist active for provider %r (%d address(es))",
        provider.name,
        len(allowed),
    )
    return proxy
