"""Native WhatsApp messaging tool via a whatsapp-web.js bridge.

Registers three LLM-callable tools:
- ``whatsapp_send``     -- send a message to a contact by name or number
- ``whatsapp_read``     -- read recent messages from a contact (default last 10)
- ``whatsapp_contacts`` -- list all contacts

All three shell out to a Node.js bridge script (whatsapp-web.js + puppeteer)
that owns the persistent WhatsApp Web auth session.  The bridge, its auth
session, and its Chrome build all live on the mounted ``/opt/data`` volume so
they survive container rebuilds.

A cold call costs ~90 s (Chrome startup + a 5 s server-ack wait), so the
subprocess timeout is 90 s.
"""

import json
import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)

# Bridge script + its bundled Chrome live on the mounted /opt/data volume.
_BRIDGE_SCRIPT = "/opt/data/whatsapp/whatsapp.js"
_CHROME_PATH = (
    "/opt/data/home/.cache/puppeteer/chrome/"
    "linux-146.0.7680.31/chrome-linux64/chrome"
)
# Cold start = Chrome boot + 5 s server-ack wait; 90 s leaves headroom.
_BRIDGE_TIMEOUT_SECONDS = 90


def _check_whatsapp_available() -> bool:
    """Toolset is only available when the Node.js bridge script is present."""
    return os.path.isfile(_BRIDGE_SCRIPT)


def _run_bridge(*args: str) -> str:
    """Invoke ``node whatsapp.js <args...>``, parse its JSON stdout, return a JSON string.

    The bridge emits a single JSON document on stdout.  A timeout, non-zero
    exit, empty output, or non-JSON output is surfaced as a tool error.
    Args are passed as a list (no shell), so contact names / messages need
    no quoting and cannot inject shell commands.
    """
    cmd = ["node", _BRIDGE_SCRIPT, *args]
    # Pin Chrome explicitly: the gateway subprocess doesn't inherit the
    # interactive shell env where the bridge is normally exercised, so
    # puppeteer's auto-discovery can't be relied on here.
    env = {**os.environ, "PUPPETEER_EXECUTABLE_PATH": _CHROME_PATH}
    # start_new_session puts `node` in its own process group so a timeout can
    # kill the whole tree.  node spawns a Chrome process tree via puppeteer;
    # killing only `node` orphans Chrome, which keeps the WhatsApp Web
    # SingletonLock held and silently blocks every subsequent run.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError:
        return tool_error("WhatsApp bridge unavailable: `node` not found on PATH")

    try:
        stdout, stderr = proc.communicate(timeout=_BRIDGE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        return tool_error(
            f"WhatsApp bridge timed out after {_BRIDGE_TIMEOUT_SECONDS}s "
            "(Chrome may have failed to start; process tree killed)"
        )

    if proc.returncode != 0:
        stderr = (stderr or "").strip()
        return tool_error(
            f"WhatsApp bridge exited {proc.returncode}: {stderr or 'no stderr'}"
        )

    stdout = (stdout or "").strip()
    if not stdout:
        return tool_error("WhatsApp bridge produced no output")

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return tool_error(
            f"WhatsApp bridge returned non-JSON output: {stdout[:500]}"
        )

    return tool_result(parsed)


# ---------------------------------------------------------------------------
# Handlers  (handler signature: (args, **kw) -> str)
# ---------------------------------------------------------------------------

def _handle_send(args: dict, **kw) -> str:
    """Handler for whatsapp_send tool."""
    contact = str(args.get("contact", "")).strip()
    message = str(args.get("message", "")).strip()
    if not contact:
        return tool_error("Missing required parameter: contact")
    if not message:
        return tool_error("Missing required parameter: message")
    return _run_bridge("send", contact, message)


def _handle_read(args: dict, **kw) -> str:
    """Handler for whatsapp_read tool."""
    contact = str(args.get("contact", "")).strip()
    if not contact:
        return tool_error("Missing required parameter: contact")
    limit = args.get("limit", 10)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error(f"Invalid 'limit' (must be an integer): {limit!r}")
    if limit < 1:
        return tool_error("'limit' must be >= 1")
    return _run_bridge("read", contact, str(limit))


def _handle_contacts(args: dict, **kw) -> str:
    """Handler for whatsapp_contacts tool."""
    return _run_bridge("contacts")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

WHATSAPP_SEND_SCHEMA = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact. The contact can be identified "
        "by display name (as saved in WhatsApp) or by phone number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": (
                    "Contact display name (e.g. 'Jane Doe') or phone number "
                    "to send the message to."
                ),
            },
            "message": {
                "type": "string",
                "description": "The message text to send.",
            },
        },
        "required": ["contact", "message"],
    },
}

WHATSAPP_READ_SCHEMA = {
    "name": "whatsapp_read",
    "description": (
        "Read recent messages from a WhatsApp conversation with a contact, "
        "identified by display name or phone number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": (
                    "Contact display name (e.g. 'Jane Doe') or phone number "
                    "whose conversation to read."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Number of most recent messages to return. Defaults to 10."
                ),
            },
        },
        "required": ["contact"],
    },
}

WHATSAPP_CONTACTS_SCHEMA = {
    "name": "whatsapp_contacts",
    "description": (
        "List all WhatsApp contacts, with their display names and numbers. "
        "Use this to resolve a name before sending or reading messages."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error, tool_result

registry.register(
    name="whatsapp_send",
    toolset="whatsapp",
    schema=WHATSAPP_SEND_SCHEMA,
    handler=_handle_send,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)

registry.register(
    name="whatsapp_read",
    toolset="whatsapp",
    schema=WHATSAPP_READ_SCHEMA,
    handler=_handle_read,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)

registry.register(
    name="whatsapp_contacts",
    toolset="whatsapp",
    schema=WHATSAPP_CONTACTS_SCHEMA,
    handler=_handle_contacts,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)
