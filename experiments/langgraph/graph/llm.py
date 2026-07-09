"""
LangChain ChatOpenAI 封装 — 替代手写 llm.py

从手写项目的 llm_config.json 读取配置，创建 ChatOpenAI 实例。
支持多 provider 切换（deepseek / openai / ollama / custom）。
"""

import json
import os
from typing import Optional
from langchain_openai import ChatOpenAI

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_config.json")

_CONFIG: Optional[dict] = None


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def get_llm(provider: Optional[str] = None) -> ChatOpenAI:
    """
    从 llm_config.json 读取配置，返回 ChatOpenAI 实例。

    参数:
        provider: provider 名称，默认从 LLM_PROVIDER 环境变量或配置中的 default 读取
    """
    config = _load_config()
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", config.get("default", "deepseek"))

    providers = config.get("providers", {})
    p = providers.get(provider)
    if p is None:
        available = ", ".join(providers.keys())
        raise ValueError(f"未知 provider: '{provider}'。可用: {available}")

    base_url = p.get("base_url", "").rstrip("/")
    api_key_env = p.get("api_key_env", "")
    direct_key = p.get("api_key", "")
    model_env = p.get("model_env", "")
    direct_model = p.get("model", "")

    api_key = os.environ.get(api_key_env, direct_key) if api_key_env else direct_key
    model = os.environ.get(model_env, direct_model) if model_env else direct_model

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=p.get("temperature", 0.7),
        max_tokens=p.get("max_tokens", 2000),
    )


def get_llm_with_tools(provider: Optional[str] = None, tools: Optional[list] = None):
    """返回绑定了工具的 ChatOpenAI 实例"""
    llm = get_llm(provider)
    if tools:
        return llm.bind_tools(tools)
    return llm
