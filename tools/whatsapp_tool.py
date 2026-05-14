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
import tempfile

logger = logging.getLogger(__name__)

# Bridge script + its bundled Chrome live on the mounted /opt/data volume.
_BRIDGE_DIR = "/opt/data/whatsapp"
_BRIDGE_SCRIPT = "/opt/data/whatsapp/whatsapp.js"
_CHROME_PATH = (
    "/opt/data/home/.cache/puppeteer/chrome/"
    "linux-146.0.7680.31/chrome-linux64/chrome"
)
# The bridge's whatsapp-web.js auth/session dir — used both by the bridge and
# to hunt down a leaked Chrome (puppeteer detaches Chrome into its own session,
# so killing `node` never reaps it).
_SESSION_DIR_MARKER = "/opt/data/whatsapp/.wwebjs_auth"
# Cold start = Chrome boot + 5 s server-ack wait; 90 s leaves headroom.
_BRIDGE_TIMEOUT_SECONDS = 90


def _check_whatsapp_available() -> bool:
    """Toolset is only available when the Node.js bridge script is present."""
    return os.path.isfile(_BRIDGE_SCRIPT)


def _kill_leaked_chrome() -> None:
    """SIGKILL any Chrome bound to the bridge's WhatsApp Web session dir.

    puppeteer launches Chrome detached (its own session), so killing the
    `node` process never reaps it.  A leaked Chrome keeps the WhatsApp Web
    SingletonLock held and silently blocks every later run, so on timeout it
    has to be hunted down by its ``--user-data-dir``.
    """
    import glob

    marker = _SESSION_DIR_MARKER.encode()
    for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(cmdline_path, "rb") as fh:
                argv = fh.read()
        except OSError:
            continue
        if marker in argv:
            try:
                os.kill(int(cmdline_path.split("/")[2]), signal.SIGKILL)
            except (OSError, ValueError):
                pass


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

    # Three non-obvious traps, all learned the hard way:
    #  - Run with cwd=_BRIDGE_DIR.  whatsapp-web.js's webVersionCache defaults
    #    to ``./.wwebjs_cache`` *relative to cwd*, and the bridge doesn't
    #    override it — run from anywhere else and client init hangs forever.
    #  - Redirect to temp files, NOT pipes.  puppeteer's Chrome inherits this
    #    process's stdout/stderr fds; with PIPE the write end stays open after
    #    `node` exits (detached Chrome holds a copy) so a pipe read hangs.
    #  - Do NOT pass start_new_session.  Running `node` as a session leader
    #    makes the puppeteer/Chrome launch itself hang indefinitely.
    out_f = tempfile.TemporaryFile(mode="w+")
    err_f = tempfile.TemporaryFile(mode="w+")
    try:
        try:
            proc = subprocess.Popen(
                cmd, stdout=out_f, stderr=err_f, env=env, cwd=_BRIDGE_DIR
            )
        except FileNotFoundError:
            return tool_error("WhatsApp bridge unavailable: `node` not found on PATH")

        try:
            proc.wait(timeout=_BRIDGE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            _kill_leaked_chrome()
            return tool_error(
                f"WhatsApp bridge timed out after {_BRIDGE_TIMEOUT_SECONDS}s "
                "(node + leaked Chrome killed)"
            )

        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().strip()
        stderr = err_f.read().strip()
    finally:
        out_f.close()
        err_f.close()

    if proc.returncode != 0:
        return tool_error(
            f"WhatsApp bridge exited {proc.returncode}: {stderr or 'no stderr'}"
        )
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
