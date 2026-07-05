"""
LLM 调用封装 — 多 Provider 支持、配置驱动、零代码改动切换模型

用法：
    from llm import LLM
    llm = LLM(provider="deepseek")      # 或 "openai" / "ollama" / "custom"
    reply = llm.chat(messages, tool_defs=...)

配置：
    默认从 llm_config.json 读取 provider 定义。
    可通过 LLM_PROVIDER 环境变量覆盖当前使用的 provider。

CLI 切换：
    export LLM_PROVIDER=openai
    python react_loop.py "你好"

    export LLM_PROVIDER=ollama
    python react_loop.py "你好"

自定义 API（通过环境变量）：
    export LLM_PROVIDER=custom
    export LLM_BASE_URL=https://api.xxx.com/v1
    export LLM_API_KEY=sk-xxx
    export LLM_MODEL=gpt-4o-mini
"""

import json
import os
import time
from typing import Optional
from urllib import request as req
from urllib.error import URLError

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")

# 全局缓存，避免多次解析
_CONFIG: Optional[dict] = None


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"LLM 配置文件不存在: {CONFIG_FILE}\n"
            f"请复制 llm_config.json 模板并填写 API Key。"
        )
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def _resolve_provider(name: str) -> dict:
    """
    解析指定 provider 的配置，返回 {
        base_url, api_key, model, temperature, max_tokens
    }
    支持环境变量覆盖：base_url_env / api_key_env / model_env
    """
    config = _load_config()
    providers = config.get("providers", {})
    if name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(
            f"未知 provider: '{name}'。可用选项: {available}\n"
            f"可通过 LLM_PROVIDER 环境变量指定，或修改 llm_config.json 中的 default 字段。"
        )

    p = dict(providers[name])  # 浅拷贝

    # 解析 api_key：优先环境变量，没有则用配置文件中直接填的 api_key
    api_key_env = p.pop("api_key_env", "")
    direct_key = p.pop("api_key", "")
    if api_key_env:
        api_key = os.environ.get(api_key_env, direct_key if direct_key else "")
    else:
        api_key = direct_key
    p["api_key"] = api_key

    # 解析 base_url：支持环境变量覆盖
    base_url_env = p.pop("base_url_env", "")
    if base_url_env:
        p["base_url"] = os.environ.get(base_url_env, p.get("base_url", ""))

    # 解析 model：支持环境变量覆盖
    model_env = p.pop("model_env", "")
    if model_env:
        p["model"] = os.environ.get(model_env, p.get("model", ""))

    # 移除描述字段（非 payload 字段）
    p.pop("description", None)

    return p


def _list_providers() -> list[str]:
    """返回所有可用的 provider 名称"""
    config = _load_config()
    return list(config.get("providers", {}).keys())


class LLM:
    """LLM 调用封装，支持任意 OpenAI 兼容 API"""

    def __init__(self, provider: Optional[str] = None):
        """
        初始化 LLM 客户端。

        参数:
            provider: provider 名称（llm_config.json 中定义）。
                      默认从 LLM_PROVIDER 环境变量读取，仍为空则用配置中的 default。
        """
        config = _load_config()
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER", config.get("default", "deepseek"))

        resolved = _resolve_provider(provider)
        self.base_url = resolved["base_url"].rstrip("/")
        self.api_key = resolved["api_key"]
        self.model = resolved["model"]
        self.temperature = resolved.get("temperature", 0.7)
        self.max_tokens = resolved.get("max_tokens", 2000)
        self.provider_name = provider

        # 检查 API Key（仅对需要 key 的 provider 检查）
        needs_key = provider != "ollama"  # Ollama 本地不需要 key
        if needs_key and not self.api_key.strip():
            print(f"[!] 没有配置 {provider} 的 API Key。")
            print(f"    方式一：设置环境变量")
            print(f"      Windows: set {config['providers'][provider]['api_key_env']}=sk-xxx")
            print(f"      Linux:   export {config['providers'][provider]['api_key_env']}=sk-xxx")
            print(f"    方式二：请在 llm_config.json 中对应的 provider 下设置 api_key 字段")
            print()

    def chat(self, messages: list, tool_defs: Optional[list] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             max_retries: int = 2) -> dict:
        """
        调用 LLM chat/completions API。

        参数:
            messages: 消息列表 [{"role": "...", "content": "..."}, ...]
            tool_defs: 工具定义列表（OpenAI Function Calling 格式）
            temperature: 覆盖配置中的 temperature
            max_tokens: 覆盖配置中的 max_tokens
            max_retries: 失败重试次数

        返回:
            LLM 返回的消息对象 {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if tool_defs is not None:
            payload["tools"] = tool_defs
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = json.dumps(payload).encode("utf-8")
        r = req.Request(url, data=body, headers=headers, method="POST")

        for attempt in range(max_retries):
            try:
                with req.urlopen(r, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result["choices"][0]["message"]
            except URLError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {"role": "assistant", "content": f"LLM调用失败: {e}"}
            except json.JSONDecodeError as e:
                return {"role": "assistant", "content": f"解析LLM返回失败: {e}"}

        return {"role": "assistant", "content": "超过最大重试次数"}

    def __repr__(self) -> str:
        return f"LLM(provider={self.provider_name}, model={self.model})"


# ===== 全局默认实例 =====
# 模块加载时自动创建，供 react_loop.py 等模块直接使用。
# 用户可通过 LLM_PROVIDER 环境变量切换默认 provider，
# 或手动创建 LLM(provider="openai") 使用不同模型。
try:
    LLM_DEFAULT = LLM()
except (FileNotFoundError, ValueError) as e:
    print(f"[警告] LLM 初始化失败: {e}")
    LLM_DEFAULT = None


def list_providers() -> list[str]:
    """列出所有可用的 provider 名称"""
    return _list_providers()
