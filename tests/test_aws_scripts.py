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
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
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
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
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
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
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
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
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
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
    seen = []
    run_permtest(World(), signed=lambda url, region, creds: seen.append(url) or 403)
    out = capsys.readouterr().out
    assert len(seen) == 3 and seen[0] == URL
    assert [l for l in out.splitlines() if "signed request" in l and l.rstrip().endswith("DENIED")]
    no_leaks(out)


def test_an_allowed_ec2_call_is_flagged_as_a_policy_mistake(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(permtest, "ROOT", tmp_path)
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
