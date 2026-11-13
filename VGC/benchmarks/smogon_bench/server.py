"""Showdown server lifecycle: start on a free port, wait until ready, kill on exit."""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from poke_env import ServerConfiguration

DEFAULT_SHOWDOWN_DIR = Path(__file__).resolve().parent / ".showdown"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_configuration(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class ShowdownServer:
    """Context manager. Pass `port` to attach to an already-running server instead."""

    def __init__(
        self,
        showdown_dir: Path = DEFAULT_SHOWDOWN_DIR,
        port: Optional[int] = None,
        log_path: Optional[Path] = None,
        ready_timeout_s: float = 90.0,
    ):
        self.dir = Path(showdown_dir)
        self.port = port
        self.attached = port is not None
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self._log_f = None

    def start(self) -> "ShowdownServer":
        if self.attached:
            if not _port_open(self.port):
                raise RuntimeError(f"nothing listening on port {self.port}")
            return self
        if not (self.dir / "pokemon-showdown").exists():
            raise RuntimeError(
                f"Showdown not found at {self.dir}; run benchmarks/smogon_bench/setup_showdown.sh"
            )
        self.port = free_port()
        out = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_f = open(self.log_path, "wb")
            out = self._log_f
        self.proc = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", str(self.port)],
            cwd=str(self.dir),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atexit.register(self.stop)
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Showdown exited early (code {self.proc.returncode})")
            if _port_open(self.port):
                return self
            time.sleep(0.2)
        self.stop()
        raise RuntimeError("Showdown did not become ready in time")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        if self._log_f:
            self._log_f.close()
            self._log_f = None

    def __enter__(self) -> "ShowdownServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def configuration(self) -> ServerConfiguration:
        return server_configuration(self.port)
