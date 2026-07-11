"""Unit tests for the optional Telegram notifier (no network; null path)."""
from detection.notifications.telegram import TelegramNotifier, _NullNotifier


def test_from_env_returns_null_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # point the loader at an empty dir so no repo .env leaks in
    monkeypatch.setattr("detection.notifications.telegram._ROOT", tmp_path)
    n = TelegramNotifier.from_env()
    assert isinstance(n, _NullNotifier)


def test_null_notifier_absorbs_all_calls():
    n = _NullNotifier()
    assert n.send_message("x") is False
    assert n.send_document("/nope") is False
    # these must not raise
    n.send_start("exp", {"arm": "zero_aug"})
    n.send_success("exp", {"ap_tail": 0.5})
    n.send_failure("exp", RuntimeError("boom"))
    n.send_separator()
