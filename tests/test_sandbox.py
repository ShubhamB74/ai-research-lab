import os

from failurelab.sandbox import Sandbox


def _data(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    return f


def test_state_persists_across_calls_and_data_is_readable(tmp_path):
    with Sandbox(_data(tmp_path)) as sb:
        assert "(2, 2)" in sb.run("import pandas as pd; df = pd.read_csv('data.csv'); print(df.shape)").stdout
        assert sb.run("print(int(df.a.sum()))").stdout.strip() == "4"


def test_workdir_contains_only_the_public_data(tmp_path):
    with Sandbox(_data(tmp_path)) as sb:
        assert sb.run("import os; print(sorted(os.listdir('.')))").stdout.strip() == "['data.csv']"


def test_secrets_in_the_parent_environment_are_not_visible(tmp_path):
    os.environ["ANTHROPIC_API_KEY"] = "sk-secret-test-value"
    try:
        with Sandbox(_data(tmp_path)) as sb:
            out = sb.run("import os; print(os.environ.get('ANTHROPIC_API_KEY')); print(sorted(os.environ))").stdout
        assert "sk-secret-test-value" not in out and out.splitlines()[0] == "None"
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_failurelab_cannot_be_imported_by_default(tmp_path):
    with Sandbox(_data(tmp_path)) as sb:
        res = sb.run("import failurelab")
    assert res.error and "not available in this sandbox" in res.error


def test_errors_and_sys_exit_do_not_kill_the_session(tmp_path):
    with Sandbox(_data(tmp_path)) as sb:
        sb.run("x = 41")
        assert "ZeroDivisionError" in sb.run("1/0").error
        assert "SystemExit" in sb.run("import sys; sys.exit(3)").error
        assert sb.run("print(x + 1)").stdout.strip() == "42"


def test_timeout_restarts_the_session_and_loses_state(tmp_path):
    with Sandbox(_data(tmp_path), timeout=1.0) as sb:
        sb.run("x = 1")
        res = sb.run("import time; time.sleep(20)")
        assert res.timed_out and "TIMEOUT" in res.as_tool_text()
        assert "NameError" in sb.run("print(x)").error          # state lost, but the sandbox still works
        assert sb.run("print('alive')").stdout.strip() == "alive"


def test_long_output_is_truncated(tmp_path):
    with Sandbox(_data(tmp_path), max_output=500) as sb:
        res = sb.run("print('x' * 100000)")
    assert res.truncated and len(res.stdout) < 800 and "truncated" in res.stdout


def test_workdir_is_removed_on_exit(tmp_path):
    with Sandbox(_data(tmp_path)) as sb:
        wd = sb.workdir
        assert wd.exists()
    assert not wd.exists()
