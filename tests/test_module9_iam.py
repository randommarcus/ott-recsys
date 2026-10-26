"""Module 9: the IAM drafts must stay least-privilege. They are never attached by this repository; these tests just read the JSON."""
import json
from pathlib import Path

IAM = Path(__file__).resolve().parents[1] / "infra" / "iam"
ROLE_ARN = "arn:aws:iam::ACCOUNT_ID:role/ott-recsys-lambda-role"
REGIONS = {"eu-central-1", "ap-south-1"}


def _load(name):
    return json.loads((IAM / name).read_text())


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


def _allows(policy):
    return [s for s in policy["Statement"] if s["Effect"] == "Allow"]


def test_no_wildcard_actions_or_resources_in_any_allow():
    for name in ("ott-dev-policy.json", "lambda-exec-role-policy.json"):
        for st in _allows(_load(name)):
            assert not any("*" in a for a in _as_list(st["Action"])), (name, st["Sid"])
            assert not any(r == "*" for r in _as_list(st["Resource"])), (name, st["Sid"])


def test_every_allowed_resource_is_an_ott_resource_in_the_two_regions():
    for name in ("ott-dev-policy.json", "lambda-exec-role-policy.json"):
        for st in _allows(_load(name)):
            for r in _as_list(st["Resource"]):
                assert "ott" in r, (name, r)
                if not r.startswith("arn:aws:iam::"):
                    assert any(f":{reg}:" in r for reg in REGIONS), (name, r)


def test_ott_dev_has_no_iam_create_or_attach_only_pass_role_on_one_arn():
    p = _load("ott-dev-policy.json")
    iam_allowed = [a for st in _allows(p) for a in _as_list(st["Action"]) if a.startswith("iam:")]
    assert iam_allowed == ["iam:PassRole"]
    pass_stmt = [st for st in _allows(p) if "iam:PassRole" in _as_list(st["Action"])][0]
    assert _as_list(pass_stmt["Resource"]) == [ROLE_ARN]
    assert pass_stmt["Condition"]["StringEquals"]["iam:PassedToService"] == "lambda.amazonaws.com"
    deny_actions = {a for st in p["Statement"] if st["Effect"] == "Deny" for a in _as_list(st.get("Action", []))}
    assert {"iam:Create*", "iam:Attach*", "iam:Put*"} <= deny_actions


def test_ott_dev_denies_other_regions_and_costly_services():
    p = _load("ott-dev-policy.json")
    deny = {s["Sid"]: s for s in p["Statement"] if s["Effect"] == "Deny"}
    cond = deny["DenyOutsideTheTwoRegions"]["Condition"]["StringNotEquals"]["aws:RequestedRegion"]
    assert set(cond) == REGIONS
    assert "ec2:*" in deny["DenyCostlyServices"]["Action"]
    assert "lambda:PutProvisionedConcurrencyConfig" in deny["DenyCostlyServices"]["Action"]


def test_execution_role_can_only_reach_its_own_things():
    p = _load("lambda-exec-role-policy.json")
    actions = {a for st in _allows(p) for a in _as_list(st["Action"])}
    assert actions == {"logs:CreateLogStream", "logs:PutLogEvents", "dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:UpdateItem", "ssm:GetParameter"}
    for st in _allows(p):
        for r in _as_list(st["Resource"]):
            if ":ssm:" in r:
                assert r.endswith("parameter/ott/zilliz")  # the ONE parameter
            if ":dynamodb:" in r:
                assert r.endswith("table/ott-users") or r.endswith("table/ott-bandit")
            if ":logs:" in r:
                assert "log-group:/aws/lambda/ott-recsys-serve:" in r
    # users table is read-only for the function; only the bandit table may be written
    users_actions = {a for st in _allows(p) for a in _as_list(st["Action"]) if any(r.endswith("table/ott-users") for r in _as_list(st["Resource"]))}
    assert users_actions == {"dynamodb:GetItem", "dynamodb:BatchGetItem"}


def test_trust_policy_only_lets_lambda_assume_the_role():
    t = _load("lambda-trust-policy.json")
    assert len(t["Statement"]) == 1
    st = t["Statement"][0]
    assert st["Principal"] == {"Service": "lambda.amazonaws.com"} and st["Action"] == "sts:AssumeRole"


def test_no_real_account_id_or_key_in_the_drafts():
    import re
    for f in IAM.glob("*.json"):
        text = f.read_text()
        assert "ACCOUNT_ID" in text or f.name == "lambda-trust-policy.json"
        assert not re.search(r"\b\d{12}\b", text), f.name
        assert "AKIA" not in text
