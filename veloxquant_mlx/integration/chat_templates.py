"""Chat-template compatibility patches owned by VeloxQuant's launchers."""

from __future__ import annotations

from typing import Any

from jinja2.exceptions import TemplateError

# Mistral-7B-Instruct-v0.3 has no dedicated system token.  This template
# accepts one initial OpenAI system message, but renders it inside the first
# instruction block -- the representation the model was trained to consume.
MISTRAL_INITIAL_SYSTEM_TEMPLATE = (
    "{{ bos_token }}"
    "{% set system_message = messages[0]['content'] if messages and messages[0]['role'] == "
    "'system' else '' %}"
    "{% set chat_messages = messages[1:] if system_message else messages %}"
    "{% for message in chat_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ ' [INST] ' + (system_message + '\n\n' if loop.first and system_message else '') + "
    "message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'] + ' ' + eos_token }}"
    "{% else %}"
    "{{ raise_exception('Only an initial system role plus user and assistant roles are supported!') }}"
    "{% endif %}{% endfor %}"
)


def ensure_initial_system_prompt_support(tokenizer: Any, model_id: str) -> bool:
    """Install a Mistral template that accepts an initial OpenAI system role.

    Returns ``True`` only when a template was changed. Templates that already
    render a leading system message (such as Qwen3.5's) are intentionally left
    untouched.
    """
    probe = [
        {"role": "system", "content": "system probe"},
        {"role": "user", "content": "user probe"},
    ]
    try:
        tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        return False
    except (TemplateError, ValueError):
        # Strict alternation is a template property, not a model-ID property.
        # Applying this template also supports Mistral-derived fine-tunes
        # whose repository ID does not include the word "mistral".
        pass

    tokenizer.chat_template = MISTRAL_INITIAL_SYSTEM_TEMPLATE
    # Fail at startup with a useful error if a future tokenizer no longer
    # accepts this Mistral-compatible Jinja syntax.
    try:
        tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    except (TemplateError, ValueError) as exc:
        raise ValueError(
            f"{model_id!r} rejects a leading system message after the "
            "VeloxQuant compatibility template was installed."
        ) from exc
    return True


__all__ = ["MISTRAL_INITIAL_SYSTEM_TEMPLATE", "ensure_initial_system_prompt_support"]
