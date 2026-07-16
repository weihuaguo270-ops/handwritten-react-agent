"""Windows/GBK-safe console helpers."""
from react_agent.console_io import FAIL, PASS, sanitize, safe_print


def test_sanitize_replaces_emoji_for_gbk():
    text = sanitize("✅ ok ❌ bad ⚠️ warn", encoding="gbk")
    assert PASS in text
    assert FAIL in text
    assert "✅" not in text
    assert "❌" not in text
    # must be encodable as GBK
    text.encode("gbk")


def test_safe_print_does_not_raise_on_emoji(capsys):
    safe_print("✅ hello")
    out = capsys.readouterr().out
    assert PASS in out or "hello" in out
