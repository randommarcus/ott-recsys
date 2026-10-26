"""Checkpoint-2 scripts, tested against a fake AWS (tests/aws_fakes.py). Nothing here touches the network, a profile or .env."""
import json

import pytest

from .aws_fakes import ACCOUNT, URL, World, client_error, load_script

permtest = load_script("aws_permtest")
teardown = load_script("aws_teardown")
put_ssm = load_script("put_ssm_secrets")
NEVER_PRINTED = (ACCOUNT, "arn:", "lambda-url", "abcdef123xyz", "REQ-ID-SHOULD-NEVER-PRINT")


def run_permtest(world, region="eu-central-1", signed=lambda url, region, creds: 200):
    return permtest.main(["--region", region], session_factory=lambda p, r: world.session(p, r), signed_call=signed, sleep=lambda s: None)


def no_leaks(text):
    for bad in NEVER_PRINTED:
        assert bad not in text, bad


# ---------------------------------------------------------------------------------------------------- aws_permtest
def test_permtest_all_allowed_cleans_up_and_prints_only_words(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    assert run_permtest(w) == 0
    out = capsys.readouterr().out
    no_leaks(out)
    body = [l for l in out.splitlines() if l.startswith("  ") and "cleanup:" not in l]
    assert body and all(("ALLOWED" in l or "DENIED (as designed)" in l) for l in body), out
    assert [l for l in out.splitlines() if "ec2 describe_instances" in l and l.rstrip().endswith("DENIED (as designed)")]
    assert w.res == set(), "every throwaway resource must be removed"
    assert w.profile == "ott-dev"
    saved = json.loads((tmp_path / "logs" / "permtest_eu-central-1.json").read_text())
    done, total = saved["cleanup"].split("/")
    assert done == total and int(total) >= 5
    no_leaks(json.dumps(saved))


def test_service_control_policy_denial_is_named_and_dependents_are_skipped(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World(deny={("lambda", "create_function"): ("AccessDeniedException", "with an explicit deny in a service control policy: arn:aws:organizations::123456789012:policy/x")})
    assert run_permtest(w) == 0
    out = capsys.readouterr().out
    no_leaks(out)
    assert "lambda create_function (128 MB, passes the role)" in out and "DENIED (service control policy)" in out
    for dependent in ("lambda invoke", "lambda create_function_url_config (AWS_IAM)", "lambda function URL: signed request"):
        assert [l for l in out.splitlines() if dependent in l and "SKIPPED" in l], dependent
    assert w.res == set()


def test_plain_and_identity_denials_are_distinguished():
    assert permtest.classify(client_error("AccessDeniedException", "User is not authorized")) == "DENIED"
    assert permtest.classify(client_error("AccessDenied", "with an explicit deny in an identity-based policy")) == "DENIED (explicit deny)"
    assert permtest.classify(client_error("UnauthorizedOperation")) == "DENIED"
    assert permtest.classify(client_error("ThrottlingException")) == "ERROR(ThrottlingException)"
    assert permtest.classify(RuntimeError("secret text")) == "ERROR(RuntimeError)"


def test_cleanup_runs_even_if_the_run_is_interrupted(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    w = World()
    real = w.session

    class Interrupt(Exception):
        pass

    orig = w.deny
    w.deny = {**orig}
    session = real()
    inv = session.client("lambda", region_name="eu-central-1")

    def boom(**kw):
        raise KeyboardInterrupt
    # interrupt at the Lambda invoke step (a BaseException is not swallowed by the per-step handler)
    from . import aws_fakes
    original_getattr = aws_fakes.Client.__getattr__

    def patched(self, method):
        if method == "invoke":
            return boom
        return original_getattr(self, method)
    monkeypatch.setattr(aws_fakes.Client, "__getattr__", patched)
    with pytest.raises(KeyboardInterrupt):
        run_permtest(w)
    assert w.res == set(), "interrupted run must still delete what it created"
    assert "cleanup:" in capsys.readouterr().out


def test_only_ott_names_are_ever_used_and_the_role_is_the_one_role(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    run_permtest(w)
    names = []
    for _, _, method, kw in w.calls:
        for k in ("FunctionName", "TableName", "logGroupName", "Name"):
            if k in kw:
                names.append(kw[k])
        names += list(kw.get("RequestItems", {}))
    assert names and all("ott" in n for n in names), names
    role = [kw["Role"] for _, _, m, kw in w.calls if m == "create_function"][0]
    assert role == f"arn:aws:iam::{ACCOUNT}:role/ott-recsys-lambda-role"
    secrets = [kw["Value"] for _, _, m, kw in w.calls if m == "put_parameter"]
    assert secrets == [permtest.DUMMY_VALUE]


def test_function_url_retries_a_fresh_403_then_reports_the_word(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    run_permtest(World(), signed=lambda url, region, creds: seen.append(url) or 403)
    out = capsys.readouterr().out
    assert len(seen) == 3 and seen[0] == URL
    assert [l for l in out.splitlines() if "signed request" in l and l.rstrip().endswith("DENIED")]
    no_leaks(out)


def test_an_allowed_ec2_call_is_flagged_as_a_policy_mistake(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    w.deny.pop(("ec2", "describe_instances"))
    run_permtest(w)
    assert "UNEXPECTED" in capsys.readouterr().out


def test_permtest_rejects_other_regions_and_reports_a_missing_profile(capsys):
    with pytest.raises(SystemExit):
        permtest.main(["--region", "us-east-1"])
    def no_profile(p, r): raise RuntimeError("profile text with " + ACCOUNT)
    assert permtest.main(["--region", "eu-central-1"], session_factory=no_profile) == 2
    out = capsys.readouterr().out
    assert "RuntimeError" in out and ACCOUNT not in out


def test_session_credentials_use_no_profile_and_skip_only_the_ec2_check(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    profiles = []
    def factory(p, r):
        profiles.append(p)
        return w.session(p, r)
    assert permtest.main(["--region", "ap-south-1", "--session-credentials"], session_factory=factory, signed_call=lambda u, r, c: 200, sleep=lambda s: None) == 0
    out = capsys.readouterr().out
    no_leaks(out)
    assert profiles == [None], "the default credential chain means profile_name=None"
    assert "credentials: default chain (no profile)" in out
    assert [l for l in out.splitlines() if "ec2 describe_instances" in l and "SKIPPED (session credentials" in l]
    assert not [c for c in w.calls if c[1] == "ec2"], "no EC2 call may be made in session mode"
    assert w.res == set() and "UNEXPECTED" not in out
    # every other step still ran and was allowed
    body = [l for l in out.splitlines() if l.startswith("  ") and "cleanup:" not in l and "ec2 describe" not in l]
    assert body and all("ALLOWED" in l for l in body)
    assert (tmp_path / "logs" / "permtest_ap-south-1.json").exists()


def test_session_credentials_keep_the_naming_rule_and_the_role(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    permtest.main(["--region", "eu-central-1", "--session-credentials"], session_factory=lambda p, r: w.session(p, r), signed_call=lambda u, r, c: 200, sleep=lambda s: None)
    names = [kw[k] for _, _, _, kw in w.calls for k in ("FunctionName", "TableName", "logGroupName", "Name") if k in kw]
    assert names and all("ott" in n for n in names)
    assert [kw["Role"] for _, _, m, kw in w.calls if m == "create_function"] == [f"arn:aws:iam::{ACCOUNT}:role/ott-recsys-lambda-role"]


def test_session_credentials_and_profile_cannot_be_combined():
    with pytest.raises(SystemExit):
        permtest.main(["--region", "eu-central-1", "--session-credentials", "--profile", "ott-dev"])
    with pytest.raises(SystemExit):
        teardown.main(["--session-credentials", "--profile", "ott-dev"])


def test_default_is_still_the_ott_dev_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = World()
    run_permtest(w)
    assert w.profile == "ott-dev"
    w2 = World()
    run_teardown(w2)
    assert w2.profile == "ott-dev"


def test_teardown_with_session_credentials(capsys):
    w = World(existing=populated())
    seen = []
    assert teardown.main(["--session-credentials", "--region", "eu-central-1", "--confirm", "DELETE-OTT"],
                         session_factory=lambda p, r: seen.append(p) or w.session(p, r)) == 0
    out = capsys.readouterr().out
    assert seen == [None] and "credentials: default chain (no profile)" in out and "remaining: 0" in out
    assert len([1 for r, k, n in w.res if r == "ap-south-1"]) == 6
    no_leaks(out)


def test_session_mode_missing_credentials_message_has_no_error_text(capsys):
    def broken(p, r): raise RuntimeError("NoCredentialsError text with " + ACCOUNT)
    assert permtest.main(["--region", "eu-central-1", "--session-credentials"], session_factory=broken) == 2
    out = capsys.readouterr().out
    assert "default credentials" in out and "RuntimeError" in out and ACCOUNT not in out


def test_cloudshell_scripts_are_standalone_single_files():
    """Upload one file: only the standard library plus boto3/botocore (imported lazily), no project code, no dotenv, no repo paths."""
    import ast, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/aws_permtest.py", "scripts/aws_teardown.py"):
        tree = ast.parse((root / name).read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            if isinstance(n, ast.ImportFrom):
                assert n.level == 0, name
                mods.add(n.module.split(".")[0])
        assert mods <= set(sys.stdlib_module_names) | {"boto3", "botocore"}, (name, mods)
        assert "__file__" not in (root / name).read_text(), f"{name} must not depend on where it sits in the repo"


# ---------------------------------------------------------------------------------------------------- aws_teardown
def populated():
    res = set()
    for r in ("eu-central-1", "ap-south-1"):
        res |= {(r, "fn", "ott-recsys-serve"), (r, "url", "ott-recsys-serve"), (r, "log", "/aws/lambda/ott-recsys-serve"), (r, "table", "ott-users"),
                (r, "table", "ott-bandit"), (r, "param", "/ott/zilliz")}
    return res


def run_teardown(world, *argv):
    return teardown.main(list(argv), session_factory=lambda p, r: world.session(p, r))


def test_teardown_dry_run_changes_nothing(capsys):
    w = World(existing=populated())
    assert run_teardown(w) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and out.count("exists") == 12 and "existing: 6" in out
    assert not [c for c in w.calls if c[2].startswith("delete")], "a dry run must not call any delete"
    assert len(w.res) == 12
    no_leaks(out)


def test_teardown_clean_world_says_existing_zero(capsys):
    assert run_teardown(World()) == 0
    out = capsys.readouterr().out
    assert out.count("existing: 0") == 2 and "unverified" not in out


def test_teardown_counts_failed_checks_as_unverified_not_as_zero(capsys):
    """If the checks themselves are denied (e.g. by a service control policy), the script must NOT report 'existing: 0'."""
    probes = {("lambda", "get_function"), ("lambda", "get_function_url_config"), ("logs", "describe_log_streams"), ("dynamodb", "describe_table"), ("ssm", "get_parameter")}
    w = World(deny={p: ("AccessDeniedException", "explicit deny in a service control policy") for p in probes})
    assert run_teardown(w, "--region", "eu-central-1") == 1
    out = capsys.readouterr().out
    assert "unverified: 11" in out and "existing: 0" not in out
    assert out.count("unknown(AccessDeniedException)") == 11
    no_leaks(out)


def test_teardown_partial_failure_is_still_unverified(capsys):
    w = World(existing=populated(), deny={("dynamodb", "describe_table"): ("ThrottlingException", "slow down")})
    assert run_teardown(w, "--region", "eu-central-1") == 1
    out = capsys.readouterr().out
    assert "unverified: 3" in out and "existing: 4" in out  # the 3 table checks failed; function, URL, log group and parameter exist
    assert "existing: 0" not in out


def test_teardown_delete_mode_with_unverifiable_checks_exits_nonzero(capsys):
    w = World(existing=populated(), deny={("dynamodb", "describe_table"): ("AccessDeniedException", "nope")})
    assert run_teardown(w, "--confirm", "DELETE-OTT", "--region", "eu-central-1") == 1
    out = capsys.readouterr().out
    assert "unverified: 3" in out


def test_teardown_needs_the_exact_confirmation_word(capsys):
    w = World(existing=populated())
    assert run_teardown(w, "--confirm", "yes") == 2
    assert w.calls == [] and len(w.res) == 12


def test_teardown_deletes_in_a_safe_order_and_rechecks(capsys):
    w = World(existing=populated())
    assert run_teardown(w, "--confirm", "DELETE-OTT") == 0
    out = capsys.readouterr().out
    assert w.res == set() and "remaining: 0" in out
    no_leaks(out)
    order = [m for r, s, m, kw in w.calls if r == "eu-central-1" and m.startswith("delete")]
    assert order.index("delete_function_url_config") < order.index("delete_function") < order.index("delete_log_group") < order.index("delete_table") < order.index("delete_parameter")


def test_teardown_reports_failures_and_exits_nonzero(capsys):
    w = World(existing=populated(), deny={("dynamodb", "delete_table"): ("AccessDeniedException", "nope")})
    assert run_teardown(w, "--confirm", "DELETE-OTT", "--region", "eu-central-1") == 1
    out = capsys.readouterr().out
    assert "FAILED(AccessDeniedException)" in out and "remaining: 2" in out  # the two tables that could not be deleted
    assert len([1 for r, k, n in w.res if r == "ap-south-1"]) == 6, "the other region must not be touched"


def test_teardown_never_touches_names_outside_ott():
    w = World(existing=populated())
    run_teardown(w, "--confirm", "DELETE-OTT")
    for _, _, _, kw in w.calls:
        for k in ("FunctionName", "TableName", "logGroupName", "Name"):
            if k in kw:
                assert "ott" in kw[k]


# ---------------------------------------------------------------------------------------------------- put_ssm_secrets
CANARY_URI, CANARY_TOKEN, FULL_TOKEN = "https://CANARY-URI-7f3a9c.invalid:19530", "CANARY-RO-TOKEN-91c2d4e8", "CANARY-FULL-TOKEN-5b6e"
ENV = {"ZILLIZ_URI": CANARY_URI, "ZILLIZ_TOKEN_READONLY": CANARY_TOKEN, "ZILLIZ_TOKEN": FULL_TOKEN}


def leaked(text):
    return [s for s in (CANARY_URI, CANARY_TOKEN, FULL_TOKEN, "7f3a9c", "91c2d4e8", "5b6e") if s in text]


def test_put_ssm_writes_one_secure_parameter_and_prints_no_value(capsys):
    w = World()
    assert put_ssm.main(["--region", "eu-central-1"], session_factory=lambda p, r: w.session(p, r), env=ENV) == 0
    out = capsys.readouterr().out
    assert out.strip() == "stored: yes (/ott/zilliz in eu-central-1)" and not leaked(out)
    (call,) = [kw for _, _, m, kw in w.calls if m == "put_parameter"]
    assert call["Name"] == "/ott/zilliz" and call["Type"] == "SecureString" and call["Overwrite"] is True
    assert json.loads(call["Value"]) == {"uri": CANARY_URI, "token": CANARY_TOKEN}
    assert FULL_TOKEN not in call["Value"]


@pytest.mark.parametrize("env", [{**ENV, "ZILLIZ_TOKEN_READONLY": ""}, {**ENV, "ZILLIZ_URI": ""}, {**ENV, "ZILLIZ_TOKEN_READONLY": FULL_TOKEN},
                                 {"ZILLIZ_URI": CANARY_URI}])
def test_put_ssm_refuses_blank_or_full_access_tokens_without_calling_aws(capsys, env):
    def never(p, r): raise AssertionError("AWS must not be contacted")
    assert put_ssm.main(["--region", "eu-central-1"], session_factory=never, env=env) == 1
    out = capsys.readouterr().out
    assert out.startswith("stored: no") and not leaked(out)


def test_put_ssm_error_paths_print_only_the_error_code(capsys):
    w = World(deny={("ssm", "put_parameter"): ("AccessDeniedException", f"not allowed {CANARY_TOKEN} {CANARY_URI}")})
    assert put_ssm.main(["--region", "ap-south-1"], session_factory=lambda p, r: w.session(p, r), env=ENV) == 1
    out = capsys.readouterr().out
    assert out.strip() == "stored: no (error: AccessDeniedException)" and not leaked(out)
    def broken(p, r): raise RuntimeError(f"profile problem {CANARY_TOKEN}")
    assert put_ssm.main(["--region", "ap-south-1"], session_factory=broken, env=ENV) == 1
    out = capsys.readouterr().out
    assert "RuntimeError" in out and not leaked(out)


def test_put_ssm_does_not_read_the_env_file_when_values_are_injected():
    def loader(): raise AssertionError("must not load .env when env is given")
    w = World()
    assert put_ssm.main(["--region", "eu-central-1"], session_factory=lambda p, r: w.session(p, r), env=ENV, env_loader=loader) == 0
