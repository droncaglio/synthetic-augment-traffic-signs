"""Telegram notifications for remote experiment monitoring.

Ported from Paper 1 (synthetic-allocation-derma/src/notifications/telegram.py),
adapted for the detection pipeline (arm terminology + AP@small / AP-tail metrics).

Configuration (via .env at the repo root, never committed):
    TELEGRAM_BOT_TOKEN=<BotFather token>
    TELEGRAM_CHAT_ID=<chat id>

Optional: if the vars are absent, from_env() returns a silent _NullNotifier and
the pipeline runs unchanged. All network calls are silent on failure — a Telegram
problem must NEVER interrupt training.
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests

_HOSTNAME = socket.gethostname()
log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TELEGRAM_DOC_API = "https://api.telegram.org/bot{token}/sendDocument"
_ROOT = Path(__file__).resolve().parents[3]  # src/detection/notifications -> repo root


def _make_log_path(label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = label.replace("/", "_").replace(" ", "_")
    logs_dir = _ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{safe}_{ts}.log"


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return False


class LogCapture:
    """Tee stdout/stderr to a log file; on exception send it to Telegram."""

    def __init__(self, log_path: Path, notifier: "TelegramNotifier", label: str = "") -> None:
        self.log_path = Path(log_path)
        self.notifier = notifier
        self.label = label
        self._orig_stdout = self._orig_stderr = self._log_file = None

    def __enter__(self) -> "LogCapture":
        import tempfile
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("w", encoding="utf-8")
        except (PermissionError, OSError) as exc:
            fallback = Path(tempfile.mktemp(suffix=".log", prefix=f"{self.log_path.stem}_"))
            log.warning(f"LogCapture: could not create {self.log_path} ({exc}). Using {fallback}")
            self.log_path = fallback
            self._log_file = self.log_path.open("w", encoding="utf-8")
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._log_file)
        sys.stderr = _Tee(self._orig_stderr, self._log_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
        try:
            self._log_file.flush()
            self._log_file.close()
        except Exception:
            pass
        if exc_type is not None:
            caption = "❌ <b>Error log</b>" + (f" — {self.label}" if self.label else "")
            try:
                self.notifier.send_document(self.log_path, caption=caption)
            except Exception:
                pass
        return False


def _load_env() -> None:
    """Load KEY=VALUE from the repo-root .env into os.environ (no overwrite)."""
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key not in os.environ:
            os.environ[key] = value


class TelegramNotifier:
    """Send formatted messages to a Telegram chat. Silent on network error."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self._api_url = _TELEGRAM_API.format(token=token)
        self._consecutive_failures = 0
        self._failure_alerted = False

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        _load_env()
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            log.info("Telegram not configured; notifications disabled.")
            return _NullNotifier()  # type: ignore[return-value]
        log.info("Telegram configured; notifications active.")
        return cls(token, chat_id)

    def send_message(self, text: str) -> bool:
        full_text = f"{text}\n<i>🖥️ {_HOSTNAME}</i>"
        try:
            resp = requests.post(
                self._api_url,
                json={"chat_id": self.chat_id, "text": full_text,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=10,
            )
            resp.raise_for_status()
            self._consecutive_failures = 0
            self._failure_alerted = False
            return True
        except requests.exceptions.RequestException as e:
            log.warning(f"Telegram: send error: {e}. Ignoring.")
            self._handle_send_failure()
            return False

    def _handle_send_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3 and not self._failure_alerted:
            print(f"\n[ALERT] Telegram disconnected — {self._consecutive_failures} "
                  f"consecutive failures.\n", flush=True)
            self._failure_alerted = True

    def send_document(self, file_path: Path, caption: str = "") -> bool:
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        if file_path.stat().st_size > 45 * 1024 * 1024:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return self.send_message(f"{caption}\n<b>(log grande — últimas 100 linhas)</b>\n"
                                     f"<pre>{chr(10).join(lines[-100:])[:3500]}</pre>")
        try:
            url = _TELEGRAM_DOC_API.format(token=self.token)
            with open(file_path, "rb") as f:
                resp = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"document": (file_path.name, f, "text/plain")}, timeout=30,
                )
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            log.warning(f"Telegram: document error: {e}. Ignoring.")
            return False

    def send_separator(self) -> None:
        self.send_message("━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def send_start(self, experiment: str, info: dict) -> None:
        smoke_tag = " <i>[SMOKE]</i>" if info.get("smoke") else ""
        self.send_message(
            f"🚀 <b>STARTING{smoke_tag}</b>\n<code>{experiment}</code>\n\n"
            f"📂 Dataset : {info.get('dataset', '?')}\n"
            f"🎨 Arm     : {info.get('arm', '?')}\n"
            f"🌱 Seed    : {info.get('seed', '?')}\n"
            f"⏱️ Epochs  : {info.get('epochs', '?')}\n"
        )

    def send_success(self, experiment: str, metrics: dict) -> None:
        hours = metrics.get("train_time_hours", 0)
        h, m = int(hours), int((hours - int(hours)) * 60)
        duration = f"{h}h{m:02d}m" if h > 0 else f"{m}min"
        self.send_message(
            f"✅ <b>COMPLETED</b>\n<code>{experiment}</code>\n\n"
            f"🪶 AP-cauda : <b>{metrics.get('ap_tail', 0):.4f}</b> <i>(macro nas tail)</i>\n"
            f"📊 AP@small : <b>{metrics.get('ap_small_macro', 0):.4f}</b> <i>(macro subset)</i>\n"
            f"📈 AP@small overall : {metrics.get('ap_small_overall', 0):.4f}\n"
            f"🔁 Epochs   : {metrics.get('epochs', '?')}\n"
            f"⏱️ Duration : {duration}\n"
        )

    def send_failure(self, experiment: str, error: Exception) -> None:
        tb_short = "\n".join(traceback.format_exc().strip().splitlines()[-3:])
        self.send_message(
            f"❌ <b>FAILED</b>\n<code>{experiment}</code>\n\n"
            f"<code>{type(error).__name__}: {str(error)[:200]}</code>\n\n<pre>{tb_short}</pre>"
        )


class _NullNotifier(TelegramNotifier):
    """Inert notifier when credentials are absent — absorbs all calls silently."""

    def __init__(self) -> None:
        pass

    def send_message(self, text: str) -> bool:
        return False

    def send_separator(self) -> None:
        pass

    def send_start(self, *a, **k) -> None:
        pass

    def send_success(self, *a, **k) -> None:
        pass

    def send_failure(self, *a, **k) -> None:
        pass

    def send_document(self, *a, **k) -> bool:
        return False
