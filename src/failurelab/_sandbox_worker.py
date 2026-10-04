"""Persistent Python worker, launched by sandbox.py as `python -I -c <this file's source>`.

Protocol: one JSON object per line on stdin ({"code": "..."}), one JSON object per line on stdout
({"stdout", "stderr", "error"}). Agent code runs in ONE shared namespace so variables persist.
Agent output is captured and never touches the protocol channel.

This is a convenience boundary for honest-but-curious agents, NOT a security boundary.
"""
import contextlib
import io
import json
import os
import sys
import traceback


def _install_import_blocker():
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "failurelab":
                raise ImportError(f"No module named {name!r} (not available in this sandbox)")
            return None
    sys.meta_path.insert(0, Block())


def main():
    if os.environ.get("FAILURELAB_ALLOW_IMPORT") == "1":      # scripted self-tests only
        src = os.environ.get("FAILURELAB_SRC")
        if src:
            sys.path.insert(0, src)
    else:
        _install_import_blocker()
    try:
        import resource
        limit = int(os.environ.get("FAILURELAB_MEM_MB", "3072")) * 2 ** 20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # not available on Windows/macOS: limit simply not applied
        pass
    out_channel = sys.__stdout__
    namespace = {"__name__": "__main__"}
    for line in sys.stdin:
        request = json.loads(line)
        stdout, stderr, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(request["code"], "<agent>", "exec"), namespace)
        except BaseException:  # includes SystemExit: the session survives agent mistakes
            error = traceback.format_exc(limit=-6)
        out_channel.write(json.dumps({"stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                                      "error": error}) + "\n")
        out_channel.flush()


if __name__ == "__main__":
    main()
