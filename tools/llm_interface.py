"""
tools/llm_interface.py — Single entry point for all LLM access.

This is the ONLY place in the codebase where LLM clients are instantiated.
All vtos, VTOS, baseline, and eval code must import from here.

Default provider: Poe  (key stored in macOS Keychain via tools.keys)
Supported providers:
  'poe'    — Poe API (OpenAI-compatible); supports gpt-* and claude-* models
  'openai' — Direct OpenAI API (fallback; requires separate key)

Usage:
    from tools.llm_interface import get_llm_client, PoeLLM, LLMInterface
    llm = get_llm_client()                          # Poe, gpt-4o-mini
    llm = get_llm_client(model="claude-sonnet-4-6") # Poe, Claude
    llm = get_llm_client(provider="openai")         # OpenAI direct
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from tools.keys import get_api_key


# ── Abstract base ─────────────────────────────────────────────────────────────

class LLMInterface(ABC):
    """Abstract base class for all LLM wrappers.

    Subclasses implement generate() which takes OpenAI-style chat messages
    and returns the assistant reply as a string.
    """

    @abstractmethod
    def generate(self, messages: List[Dict]) -> str:
        """Generate a reply for the given chat messages."""

    def complete(self, user: str, system: Optional[str] = None) -> str:
        """Convenience: single user turn with optional system prompt.

        Wraps generate() with a standard messages list. Used by VTOS's
        vision_agent.py (which expects this signature).
        """
        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.generate(messages)

    def generate_batch(self, messages_batch: List[List[Dict]], max_workers: int = 3) -> List[str]:
        """Concurrent batch generation."""
        import concurrent.futures
        results = [""] * len(messages_batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(self.generate, msgs): i
                             for i, msgs in enumerate(messages_batch)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"⚠️ Batch generation error task {idx}: {exc}")
                    results[idx] = f"Error: {exc}"
        return results

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available model IDs for this provider."""


# ── Poe (default provider) ────────────────────────────────────────────────────

class PoeLLM(LLMInterface):
    """Poe API wrapper (OpenAI-compatible endpoint).

    Supports both gpt-* and claude-* models via Poe's unified API.
    API key is retrieved from the centralised keystore (Keychain → env → file).

    Available models on Poe:
      gpt-4o-mini, gpt-4o
      claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-6
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_tokens: int = 4096):
        from openai import OpenAI
        if api_key is None:
            api_key = get_api_key("poe")
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

    def generate(self, messages: List[Dict], model: str = None) -> str:
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error calling Poe API: {e}"

    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self.client.models.list()]
        except Exception as e:
            print(f"Error listing Poe models: {e}")
            return []


# ── OpenAI direct (fallback) ──────────────────────────────────────────────────

class OpenAILLM(LLMInterface):
    """Direct OpenAI API wrapper (fallback; not the default provider)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_tokens: int = 4096):
        import httpx
        from openai import OpenAI
        if api_key is None:
            api_key = get_api_key("openai")
        self.model = model
        self.max_tokens = max_tokens
        http_client = httpx.Client(base_url="https://api.openai.com/v1", timeout=30.0)
        self.client = OpenAI(api_key=api_key, http_client=http_client)

    def generate(self, messages: List[Dict], model: str = None) -> str:
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
        for m in messages:
            if "content" not in m or not isinstance(m["content"], (str, list)):
                raise ValueError("each message must have a string or list 'content'")
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self.client.models.list()]
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return []


# ── OpenRouter (open-source VLM access) ───────────────────────────────────────

class OpenRouterLLM(LLMInterface):
    """OpenRouter API wrapper (OpenAI-compatible endpoint).

    Provides access to open-source VLMs (Qwen3-VL, Llama, InternVL, etc.)
    via a single OpenAI-compatible API. API key stored in macOS Keychain
    under service='openrouter', account='vtos'.

    Common model IDs:
      qwen/qwen3-vl-8b-instruct
      qwen/qwen3-vl-32b-instruct
      qwen/qwen3-vl-235b-a22b-instruct
      meta-llama/llama-3.2-90b-vision-instruct
    """

    def __init__(self, model: str = "qwen/qwen3-vl-8b-instruct",
                 api_key: Optional[str] = None, max_tokens: int = 4096):
        from openai import OpenAI
        if api_key is None:
            api_key = get_api_key("openrouter")
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def generate(self, messages: List[Dict], model: str = None) -> str:
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenRouter API: {e}"

    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self.client.models.list()]
        except Exception as e:
            print(f"Error listing OpenRouter models: {e}")
            return []


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client(provider: str = "poe", model: Optional[str] = None) -> LLMInterface:
    """
    Factory — the single authorised way to obtain an LLM client.

    Args:
        provider: 'poe' (default), 'openai', or 'openrouter'.
        model:    Model name. Defaults vary by provider.

    Returns:
        An LLMInterface instance ready for generate() calls.
    """
    p = provider.lower()
    if p == "poe":
        return PoeLLM(model=model or "gpt-4o-mini")
    elif p == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    elif p == "openrouter":
        return OpenRouterLLM(model=model or "qwen/qwen3-vl-8b-instruct")
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: 'poe', 'openai', 'openrouter'."
        )
