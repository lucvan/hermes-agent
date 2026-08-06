"""Arbitrary-contact WhatsApp tools on top of upstream's Baileys bridge.

Not part of upstream. Upstream's ``hermes-whatsapp`` platform is reactive
only: the agent sees messages as they arrive in whichever chat is messaging
it, and no platform gets an agent-callable send tool (see toolsets.py's
"agents do NOT get an agent-callable send_message tool" note) — outbound is
deliberately routed through cron delivery, the kanban notifier, or
``hermes send`` instead of a model-invoked tool call.

Hermes previously ran a separate whatsapp-web.js bridge with
whatsapp_send/read/contacts tools for exactly this — proactively messaging or
reading history from a contact who isn't the current chat. This restores that
on the new Baileys bridge (scripts/whatsapp-bridge/bridge.js's local
``/contacts`` and ``/history/:id`` extensions + plugins/platforms/whatsapp/
adapter.py's resolve_contact/list_contacts/get_history) instead of keeping a
second bridge process alive.

Registers three LLM-callable tools in the opt-in ``whatsapp-extras``
toolset:
- ``whatsapp_send``     -- send a message to a contact by name or number
- ``whatsapp_read``     -- read recent buffered messages from a contact
- ``whatsapp_contacts`` -- list known contacts/groups

Known limitation vs. the old whatsapp-web.js bridge: ``whatsapp_read`` is
live-forward only (messages seen since the bridge process started), not an
on-demand fetch of WhatsApp's own history — Baileys keeps no message store.
See adapter.py's get_history() docstring.
"""

import json

from tools.registry import registry, tool_error


def _check_whatsapp_available() -> bool:
    """Toolset is only available when the WhatsApp platform is configured."""
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.WHATSAPP)
        return bool(pconfig and pconfig.enabled)
    except Exception:
        return False


def _get_pconfig():
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    return config.platforms.get(Platform.WHATSAPP)


def _handle_send(args: dict, **kw) -> str:
    """Handler for whatsapp_send tool."""
    contact = str(args.get("contact", "")).strip()
    message = str(args.get("message", "")).strip()
    if not contact:
        return tool_error("Missing required parameter: contact")
    if not message:
        return tool_error("Missing required parameter: message")

    from model_tools import _run_async
    from plugins.platforms.whatsapp.adapter import resolve_contact

    pconfig = _get_pconfig()
    jid = _run_async(resolve_contact(pconfig, contact))
    if not jid:
        return tool_error(
            f"Contact not found: {contact!r}. Use whatsapp_contacts to list "
            f"known contacts, or pass a phone number directly."
        )

    # Reuse send_message_tool's mature delivery path (live adapter when the
    # gateway is running in this process, standalone bridge HTTP fallback
    # otherwise, chunking, etc.) rather than re-implementing it. The
    # resolved JID is an explicit WhatsApp target ref, so this skips
    # send_message_tool's channel-name resolution.
    from tools.send_message_tool import _handle_send as _send

    return _send({"target": f"whatsapp:{jid}", "message": message})


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

    from model_tools import _run_async
    from plugins.platforms.whatsapp.adapter import get_history, resolve_contact

    pconfig = _get_pconfig()
    jid = _run_async(resolve_contact(pconfig, contact))
    if not jid:
        return tool_error(
            f"Contact not found: {contact!r}. Use whatsapp_contacts to list "
            f"known contacts, or pass a phone number directly."
        )
    result = _run_async(get_history(pconfig, jid, limit))
    return json.dumps(result)


def _handle_contacts(args: dict, **kw) -> str:
    """Handler for whatsapp_contacts tool."""
    from model_tools import _run_async
    from plugins.platforms.whatsapp.adapter import list_contacts

    pconfig = _get_pconfig()
    result = _run_async(list_contacts(pconfig))
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

WHATSAPP_SEND_SCHEMA = {
    "name": "whatsapp_send",
    "description": (
        "Send a WhatsApp message to a contact. The contact can be identified "
        "by display name (as known from synced contacts/groups) or by phone "
        "number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": (
                    "Contact or group display name (e.g. 'Jane Doe') or phone "
                    "number to send the message to."
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
        "Read recent WhatsApp messages from a conversation with a contact, "
        "identified by display name or phone number. Only covers messages "
        "seen since the bridge last (re)started, not WhatsApp's full history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": (
                    "Contact or group display name (e.g. 'Jane Doe') or phone "
                    "number whose conversation to read."
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
        "List known WhatsApp contacts and groups, with their display names. "
        "Only includes identities WhatsApp has synced to this session or who "
        "have messaged it — not a full address-book export. Use this to "
        "resolve a name before sending or reading messages."
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

registry.register(
    name="whatsapp_send",
    toolset="whatsapp-extras",
    schema=WHATSAPP_SEND_SCHEMA,
    handler=_handle_send,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)

registry.register(
    name="whatsapp_read",
    toolset="whatsapp-extras",
    schema=WHATSAPP_READ_SCHEMA,
    handler=_handle_read,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)

registry.register(
    name="whatsapp_contacts",
    toolset="whatsapp-extras",
    schema=WHATSAPP_CONTACTS_SCHEMA,
    handler=_handle_contacts,
    check_fn=_check_whatsapp_available,
    emoji="💬",
)
