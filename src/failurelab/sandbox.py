"""Sandboxed Python execution for research agents.

What it gives you
  * a PERSISTENT Python session per task (variables survive between calls)
  * a working directory containing ONLY a copy of the public data as `data.csv`
  * a scrubbed environment: no API keys, no inherited variables (HOME points at the workdir)
  * per-call wall-clock timeout (the session is killed and restarted, state is lost)
  * output truncation, memory cap (POSIX), `failurelab` imports blocked

What it is NOT: a security boundary. A hostile model could still read other files the OS user can read
or use the network. For stronger isolation pass `cmd_prefix` (e.g. ["unshare", "-rn"] on Linux to drop
network access) or run the whole harness in a locked-down container. See docs/RUNNING_EXPERIMENTS.md.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_WORKER_SRC = (Path(__file__).parent / "_sandbox_worker.py").read_text()


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    seconds: float = 0.0
    timed_out: bool = False
    crashed: bool = False
    truncated: bool = False

    def as_tool_text(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if self.error:
            parts.append("[error]\n" + self.error.rstrip())
        if self.timed_out:
            parts.append("[TIMEOUT: the session was killed and restarted; all variables are lost]")
        if self.crashed:
            parts.append("[the Python process crashed (possibly out of memory); session restarted, variables lost]")
        return "\n".join(parts) if parts else "(no output)"


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.66)
    return text[:head] + f"\n... [{len(text) - limit} characters truncated] ...\n" + text[-(limit - head):], True


class Sandbox:
    def __init__(self, data_path: str | Path, *, timeout: float = 60.0, max_output: int = 4000,
                 mem_mb: int = 3072, allow_failurelab: bool = False, cmd_prefix: list[str] | None = None):
        self.data_path = Path(data_path)
        self.timeout, self.max_output, self.mem_mb = timeout, max_output, mem_mb
        self.allow_failurelab = allow_failurelab
        self.cmd_prefix = list(cmd_prefix or [])
        self.workdir: Path | None = None
        self.proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "Sandbox":
        self.workdir = Path(tempfile.mkdtemp(prefix="flab_sbx_"))
        shutil.copy(self.data_path, self.workdir / "data.csv")
        self._spawn()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._kill()
        if self.workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def _env(self) -> dict[str, str]:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.workdir), "LANG": "C.UTF-8",
               "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "FAILURELAB_MEM_MB": str(self.mem_mb)}
        if "SYSTEMROOT" in os.environ:   # required for Python on Windows
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        if self.allow_failurelab:
            env["FAILURELAB_ALLOW_IMPORT"] = "1"
            env["FAILURELAB_SRC"] = str(Path(__file__).resolve().parents[1])
        return env

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(
            [*self.cmd_prefix, sys.executable, "-I", "-c", _WORKER_SRC],
            cwd=self.workdir, env=self._env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self._q: queue.Queue = queue.Queue()
        proc, q = self.proc, self._q

        def reader():
            for line in iter(proc.stdout.readline, ""):
                q.put(line)
            q.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        if self.proc:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None

    # ------------------------------------------------------------------ execution
    def run(self, code: str) -> ExecResult:
        assert self.proc is not None, "use the Sandbox as a context manager"
        start = time.time()
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            line = self._q.get(timeout=self.timeout)
        except queue.Empty:
            self._kill(); self._spawn()
            return ExecResult(seconds=time.time() - start, timed_out=True)
        except (BrokenPipeError, OSError):
            line = None
        if line is None:
            self._kill(); self._spawn()
            return ExecResult(seconds=time.time() - start, crashed=True)
        reply = json.loads(line)
        out, t1 = _truncate(reply["stdout"], self.max_output)
        err, t2 = _truncate(reply["stderr"], self.max_output // 2)
        error, t3 = _truncate(reply["error"] or "", self.max_output // 2)
        return ExecResult(stdout=out, stderr=err, error=error or None, seconds=time.time() - start,
                          truncated=t1 or t2 or t3)
