"""The standalone CloudShell scripts (scripts/cloudshell/*.py) against the fake AWS. Nothing here touches the network, a profile or .env."""
import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ott_recsys.serving.records import pack, unpack

from .aws_fakes import ACCOUNT, URL, World, load_script

ROOT = Path(__file__).resolve().parents[1]
put_param = load_script("cloudshell/put_zilliz_param")
loader = load_script("cloudshell/load_dynamodb")
deploy = load_script("cloudshell/deploy_lambda")
exporter = load_script("export_dynamodb_items")
BAD = (ACCOUNT, "arn:", "lambda-url", "abcdef123xyz")
CANARY_URI, CANARY_KEY = "https://CANARY-URI-7f3a9c.invalid:19530", "CANARY-RO-KEY-91c2d4e8"


def sess(world):
    return lambda region: world.session("default-chain", region)


# ------------------------------------------------------------------------------------------------ put_zilliz_param
def scripted(*answers):
    it = iter(answers)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return next(it)
    ask.prompts = prompts
    return ask


def test_param_script_uses_hidden_prompts_and_stores_one_secure_parameter(capsys):
    w = World()
    ask = scripted(CANARY_URI, CANARY_KEY, CANARY_KEY)
    assert put_param.main(["--region", "eu-central-1"], session_factory=sess(w), ask=ask) == 0
    out = capsys.readouterr().out
    assert out.strip() == "stored: yes (/ott/zilliz in eu-central-1)" and "CANARY" not in out and "7f3a9c" not in out
    assert all("hidden" in p for p in ask.prompts) and len(ask.prompts) == 3
    (call,) = [kw for _, _, m, kw in w.calls if m == "put_parameter"]
    assert call["Name"] == "/ott/zilliz" and call["Type"] == "SecureString" and json.loads(call["Value"]) == {"uri": CANARY_URI, "token": CANARY_KEY}


@pytest.mark.parametrize("answers", [("", CANARY_KEY, CANARY_KEY), (CANARY_URI, "", ""), (CANARY_URI, CANARY_KEY, "typo-" + CANARY_KEY), ("http://plain.invalid", CANARY_KEY, CANARY_KEY)])
def test_param_script_refuses_bad_input_without_calling_aws(capsys, answers):
    def never(region): raise AssertionError("AWS must not be contacted")
    assert put_param.main(["--region", "eu-central-1"], session_factory=never, ask=scripted(*answers)) == 1
    out = capsys.readouterr().out
    assert out.startswith("stored: no") and "CANARY" not in out and "7f3a9c" not in out and "91c2d4e8" not in out


def test_param_script_errors_print_only_the_code(capsys):
    w = World(deny={("ssm", "put_parameter"): ("AccessDeniedException", f"nope {CANARY_KEY} {CANARY_URI}")})
    assert put_param.main(["--region", "ap-south-1"], session_factory=sess(w), ask=scripted(CANARY_URI, CANARY_KEY, CANARY_KEY)) == 1
    out = capsys.readouterr().out
    assert out.strip() == "stored: no (error: AccessDeniedException)"


def test_param_script_never_reads_the_secret_from_arguments_or_input():
    src = (ROOT / "scripts/cloudshell/put_zilliz_param.py").read_text()
    tree = ast.parse(src)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "input"], "input() would echo the key"
    assert "getpass" in src
    ap_args = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"]
    assert ap_args == ["--region"], "no secret may be a command-line argument"


# ------------------------------------------------------------------------------------------------ load_dynamodb
def make_files(tmp_path, n_users=30, chunks=3):
    rng = np.random.default_rng(0)
    recs = [{"user_id": 100 + i, "cand": rng.integers(-1, 50, 230).astype(np.int32), "last_ts": 1_500_000_000 + i, "sess_len": 3, "sess_likes": 1, "ncf_known": 1,
             "ttvec": rng.normal(size=64).astype(np.float32), "prof": rng.normal(size=384).astype(np.float32), "seen": np.arange(i + 1, dtype=np.int32)} for i in range(n_users)]
    edges = np.linspace(0, n_users, chunks + 1).astype(int)
    manifest = {"users": n_users, "chunks": []}
    for c in range(chunks):
        path = tmp_path / f"users_items_part{c + 1}.jsonl.gz"
        with gzip.open(path, "wt") as f:
            for r in recs[edges[c]:edges[c + 1]]:
                f.write(exporter.encode(pack(r)) + "\n")
        manifest["chunks"].append({"file": path.name, "users": int(edges[c + 1] - edges[c]), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return recs


def test_export_encoding_round_trips_to_the_same_records():
    rec = {"user_id": 5, "cand": np.array([3, -1, 7], np.int32), "last_ts": 1_500_000_000, "sess_len": 4, "sess_likes": 2, "ncf_known": 1,
           "ttvec": np.arange(4, dtype=np.float32), "seen": np.array([1, 2], np.int32)}
    back = loader.decode(exporter.encode(pack(rec)))
    assert back == pack(rec)
    assert np.array_equal(unpack(back)["ttvec"], rec["ttvec"])


def test_tables_are_created_on_demand_only_with_the_two_fixed_names(capsys):
    w = World()
    assert loader.main(["--region", "eu-central-1", "--create-tables"], session_factory=sess(w)) == 0
    assert w.res == {("eu-central-1", "table", "ott-users"), ("eu-central-1", "table", "ott-bandit")}
    creates = [kw for _, _, m, kw in w.calls if m == "create_table"]
    assert {c["TableName"] for c in creates} == {"ott-users", "ott-bandit"} and all(c["BillingMode"] == "PAY_PER_REQUEST" for c in creates)
    assert {c["KeySchema"][0]["AttributeName"] for c in creates} == {"user_id", "arm_id"}
    assert loader.main(["--region", "eu-central-1", "--create-tables"], session_factory=sess(w)) == 0  # running twice is fine
    assert "already exists" in capsys.readouterr().out


def test_tables_denied_by_a_policy_is_reported_in_words(capsys):
    w = World(deny={("dynamodb", "create_table"): ("AccessDeniedException", "with an explicit deny in a service control policy arn:aws:x:123456789012")})
    assert loader.main(["--region", "eu-central-1", "--create-tables"], session_factory=sess(w)) == 1
    out = capsys.readouterr().out
    assert "DENIED" in out and not any(b in out for b in BAD)


def test_load_verifies_hashes_writes_every_user_and_check_finds_them(tmp_path, capsys):
    recs = make_files(tmp_path)
    w = World(existing={("eu-central-1", "table", "ott-users")})
    assert loader.main(["--region", "eu-central-1", "--load", "--dir", str(tmp_path)], session_factory=sess(w)) == 0
    out = capsys.readouterr().out
    assert "loaded: 30 items into ott-users" in out and not any(b in out for b in BAD)
    stored = w.items[("eu-central-1", "ott-users")]
    assert set(stored) == {r["user_id"] for r in recs}
    assert np.array_equal(unpack(stored[105])["ttvec"], recs[5]["ttvec"])
    assert loader.main(["--region", "eu-central-1", "--check", "--dir", str(tmp_path)], session_factory=sess(w)) == 0
    assert "found" in capsys.readouterr().out


def test_load_refuses_a_file_that_does_not_match_the_manifest(tmp_path, capsys):
    make_files(tmp_path)
    with gzip.open(tmp_path / "users_items_part2.jsonl.gz", "wt") as f:
        f.write('{"user_id": 1}\n')
    w = World(existing={("eu-central-1", "table", "ott-users")})
    assert loader.main(["--region", "eu-central-1", "--load", "--dir", str(tmp_path)], session_factory=sess(w)) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert 1 not in w.items.get(("eu-central-1", "ott-users"), {})


def test_load_stops_with_a_resume_hint_when_writes_are_denied(tmp_path, capsys):
    make_files(tmp_path)
    w = World(existing={("eu-central-1", "table", "ott-users")}, deny={("dynamodb", "batch_write_item"): ("AccessDeniedException", "nope")})
    assert loader.main(["--region", "eu-central-1", "--load", "--dir", str(tmp_path)], session_factory=sess(w)) == 1
    out = capsys.readouterr().out
    assert "DENIED" in out and "--start-part 1" in out


# ------------------------------------------------------------------------------------------------ deploy_lambda
def zip_file(tmp_path):
    z = tmp_path / "lambda.zip"
    z.write_bytes(b"PK-not-a-real-zip")
    return z


def test_create_makes_log_group_function_and_an_iam_only_url(tmp_path, capsys):
    w = World()
    z = zip_file(tmp_path)
    assert deploy.main(["--region", "eu-central-1", "--zip", str(z), "create"], session_factory=sess(w)) == 0
    out = capsys.readouterr().out
    assert not any(b in out for b in BAD)
    calls = {m: kw for _, _, m, kw in w.calls}
    assert calls["put_retention_policy"]["retentionInDays"] == 7 and calls["put_retention_policy"]["logGroupName"] == "/aws/lambda/ott-recsys-serve"
    cf = calls["create_function"]
    assert cf["FunctionName"] == "ott-recsys-serve" and cf["Runtime"] == "python3.12" and cf["Handler"] == "ott_recsys.serving.handler.handler"
    assert cf["Role"] == f"arn:aws:iam::{ACCOUNT}:role/ott-recsys-lambda-role" and cf["Code"]["ZipFile"] == b"PK-not-a-real-zip"
    assert cf["MemorySize"] == 1024 and cf["Timeout"] == 10 and "Environment" not in cf, "no environment variables, so no secret can sit there"
    assert calls["create_function_url_config"]["AuthType"] == "AWS_IAM", "never a public URL"
    assert "put_provisioned_concurrency_config" not in calls
    assert [kw["ReservedConcurrentExecutions"] for _, _, m, kw in w.calls if m == "put_function_concurrency"] == [2]


def test_create_stops_at_the_first_denied_step(tmp_path, capsys):
    w = World(deny={("lambda", "create_function"): ("AccessDeniedException", "service control policy")})
    assert deploy.main(["--region", "ap-south-1", "--zip", str(zip_file(tmp_path)), "create"], session_factory=sess(w)) == 1
    out = capsys.readouterr().out
    assert "create function" in out and "DENIED" in out and "function URL" not in out


def test_create_needs_the_zip_and_names_nothing_outside_ott(tmp_path, capsys):
    w = World()
    assert deploy.main(["--region", "eu-central-1", "--zip", str(tmp_path / "missing.zip"), "create"], session_factory=sess(w)) == 2
    deploy.main(["--region", "eu-central-1", "--zip", str(zip_file(tmp_path)), "create"], session_factory=sess(w))
    for _, _, _, kw in w.calls:
        for k in ("FunctionName", "logGroupName"):
            if k in kw:
                assert "ott" in kw[k]


def test_smoke_prints_counts_and_never_the_url(tmp_path, capsys):
    w = World(existing={("eu-central-1", "fn", "ott-recsys-serve")})
    assert deploy.main(["--region", "eu-central-1", "smoke", "--user-id", "7"], session_factory=sess(w)) == 0
    out = capsys.readouterr().out
    assert "20 items, server 12.3 ms" in out and not any(b in out for b in BAD)
    bodies = [json.loads(kw["Payload"]) for _, _, m, kw in w.calls if m == "invoke"]
    assert any(json.loads(b["body"]).get("device") == "mobile" for b in bodies if b["body"]), "the M3b call must pass a SIMULATED device"
    assert all(json.loads(b["body"]).get("record_impressions", True) is False for b in bodies if json.loads(b["body"]).get("user_id")), "smoke tests must not change bandit counts"


def test_smoke_checks_live_mode_and_fails_on_a_fallback(capsys):
    w = World(existing={("eu-central-1", "fn", "ott-recsys-serve")})
    assert deploy.main(["--region", "eu-central-1", "smoke"], session_factory=sess(w)) == 0
    out = capsys.readouterr().out
    assert "LIVE (SSM key + Zilliz search)" in out and "2 Zilliz searches, search 40.0 ms" in out
    w.live_fallback = "live retrieval failed (ConfigError); used the precomputed list"
    assert deploy.main(["--region", "eu-central-1", "smoke"], session_factory=sess(w)) == 1
    assert "FELL BACK to precomputed (live retrieval failed (ConfigError)" in capsys.readouterr().out


def test_show_url_is_the_only_action_that_prints_the_url(tmp_path, capsys):
    w = World(existing={("eu-central-1", "fn", "ott-recsys-serve"), ("eu-central-1", "url", "ott-recsys-serve")})
    assert deploy.main(["--region", "eu-central-1", "show-url"], session_factory=sess(w)) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == URL


def test_update_code_uploads_the_new_zip(tmp_path, capsys):
    w = World(existing={("eu-central-1", "fn", "ott-recsys-serve")})
    assert deploy.main(["--region", "eu-central-1", "--zip", str(zip_file(tmp_path)), "update-code"], session_factory=sess(w)) == 0
    assert [kw["ZipFile"] for _, _, m, kw in w.calls if m == "update_function_code"] == [b"PK-not-a-real-zip"]


# ------------------------------------------------------------------------------------------------ all of them
CLOUDSHELL = sorted((ROOT / "scripts" / "cloudshell").glob("*.py"))


@pytest.mark.parametrize("path", CLOUDSHELL, ids=lambda p: p.name)
def test_each_cloudshell_script_is_one_standalone_file(path):
    tree = ast.parse(path.read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        if isinstance(n, ast.ImportFrom):
            assert n.level == 0
            mods.add(n.module.split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"boto3", "botocore"}, mods
    text = path.read_text()
    assert "__file__" not in text and "ott_recsys" not in text.replace("ott_recsys.serving.handler.handler", "")
    assert "profile_name" not in text and "~/.aws" not in text, "CloudShell scripts use the default credential chain only"


def test_there_are_exactly_the_expected_cloudshell_scripts():
    assert [p.name for p in CLOUDSHELL] == ["deploy_lambda.py", "load_dynamodb.py", "put_zilliz_param.py"]
