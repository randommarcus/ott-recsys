"""A tiny fake AWS used by the checkpoint-2 script tests. No network, no profile, no credentials."""
import importlib.util
from pathlib import Path

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "123456789012"
URL = "https://abcdef123xyz.lambda-url.eu-central-1.on.aws/"


def load_script(name):
    spec = importlib.util.spec_from_file_location("script_" + name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def client_error(code, message="boom", op="Op"):
    return ClientError({"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"RequestId": "REQ-ID-SHOULD-NEVER-PRINT"}}, op)


class World:
    """Existing resources per region plus a deny map {(service, method): (code, message)} and a call log."""

    def __init__(self, deny=None, existing=()):
        self.res = set(existing)  # (region, kind, name)
        self.deny = {("ec2", "describe_instances"): ("UnauthorizedOperation", "You are not authorized")}
        self.deny.update(deny or {})
        self.calls = []  # (region, service, method, kwargs)
        self.ssm_values = {}
        self.fail_after = None

    def session(self, profile="ott-dev", region="eu-central-1"):
        self.profile = profile
        return Session(self, region)


class Session:
    def __init__(self, world, region):
        self.w, self.region = world, region

    def client(self, service, region_name=None, **kw):
        return Client(self.w, service, region_name or self.region)

    def get_credentials(self):
        class C:
            def get_frozen_credentials(self): return object()
        return C()


class Waiter:
    def wait(self, **kw): return None


KINDS = {"create_log_group": ("log", "logGroupName"), "create_table": ("table", "TableName"), "create_function": ("fn", "FunctionName"),
         "put_parameter": ("param", "Name")}
DELETES = {"delete_log_group": ("log", "logGroupName"), "delete_table": ("table", "TableName"), "delete_function": ("fn", "FunctionName"),
           "delete_parameter": ("param", "Name"), "delete_function_url_config": ("url", "FunctionName")}
PROBES = {"describe_log_streams": ("log", "logGroupName"), "describe_table": ("table", "TableName"), "get_function": ("fn", "FunctionName"),
          "get_function_url_config": ("url", "FunctionName"), "get_parameter": ("param", "Name")}
MISSING = {"log": "ResourceNotFoundException", "table": "ResourceNotFoundException", "fn": "ResourceNotFoundException", "url": "ResourceNotFoundException",
           "param": "ParameterNotFound"}


class Client:
    def __init__(self, world, service, region):
        self.w, self.service, self.region = world, service, region

    def get_waiter(self, name):
        return Waiter()

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(**kw):
            w = self.w
            w.calls.append((self.region, self.service, method, kw))
            if (self.service, method) in w.deny:
                raise client_error(*w.deny[(self.service, method)], op=method)
            if method == "get_caller_identity":
                return {"Account": ACCOUNT}
            if method in KINDS:
                kind, key = KINDS[method]
                ident = (self.region, kind, kw[key])
                if ident in w.res and kind != "param":
                    raise client_error("ResourceInUseException" if kind == "table" else "ResourceAlreadyExistsException", op=method)
                w.res.add(ident)
                if kind == "param":
                    w.ssm_values[kw[key]] = kw["Value"]
                if kind == "fn":
                    return {"FunctionName": kw[key]}
                return {}
            if method in DELETES:
                kind, key = DELETES[method]
                ident = (self.region, kind, kw[key])
                if ident not in w.res:
                    raise client_error(MISSING[kind], op=method)
                w.res.discard(ident)
                return {}
            if method in PROBES:
                kind, key = PROBES[method]
                if (self.region, kind, kw[key]) not in w.res:
                    raise client_error(MISSING[kind], op=method)
                if method == "get_parameter":
                    return {"Parameter": {"Value": w.ssm_values.get(kw[key], "")}}
                return {}
            if method == "create_function_url_config":
                w.res.add((self.region, "url", kw["FunctionName"]))
                return {"FunctionUrl": URL}
            if method == "invoke":
                return {"StatusCode": 200}
            return {}
        return call
