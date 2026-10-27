"""A tiny fake AWS used by the checkpoint-2 script tests. No network, no profile, no credentials."""
import importlib.util
import io
import json
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
        self.items = {}
        self.live_fallback = None
        self.slow_table_delete = False  # DynamoDB-like: a deleted table stays visible with status DELETING
        self.deleting = set()  # set to the handler's fallback text to simulate a live-mode failure
        self.fail_after = None

    def session(self, profile="ott-dev", region="eu-central-1"):
        self.profile = profile
        return Session(self, region)


class Session:
    def __init__(self, world, region):
        self.w, self.region = world, region

    def client(self, service, region_name=None, **kw):
        return Client(self.w, service, region_name or self.region)

    def resource(self, service, region_name=None, **kw):
        return Resource(self.w, region_name or self.region)

    def get_credentials(self):
        class C:
            def get_frozen_credentials(self): return object()
        return C()


class Resource:
    """boto3 DynamoDB resource: Table(name).batch_writer / get_item over world.items[(region, table)][key]."""
    def __init__(self, world, region):
        self.w, self.region = world, region

    def Table(self, name):
        return Table(self.w, self.region, name)


class Table:
    def __init__(self, world, region, name):
        self.w, self.region, self.name = world, region, name

    def _guard(self, method):
        self.w.calls.append((self.region, "dynamodb", method, {"TableName": self.name}))
        if ("dynamodb", method) in self.w.deny:
            raise client_error(*self.w.deny[("dynamodb", method)], op=method)
        if (self.region, "table", self.name) not in self.w.res:
            raise client_error("ResourceNotFoundException", op=method)
        return self.w.items.setdefault((self.region, self.name), {})

    def batch_writer(self, overwrite_by_pkeys=None):
        table = self

        class BW:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def put_item(s, Item):
                store = table._guard("batch_write_item")
                for k, v in Item.items():
                    assert isinstance(v, (int, bytes)) and not isinstance(v, float), "only ints and bytes may be written"
                store[Item["user_id"]] = Item
        return BW()

    def get_item(self, Key):
        store = self._guard("get_item")
        item = store.get(Key["user_id"])
        return {"Item": item} if item is not None else {}


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
                if ident not in w.res or ident in w.deleting:
                    raise client_error(MISSING[kind], op=method)
                if kind == "table" and w.slow_table_delete:
                    w.deleting.add(ident)
                else:
                    w.res.discard(ident)
                return {}
            if method in PROBES:
                kind, key = PROBES[method]
                if (self.region, kind, kw[key]) not in w.res:
                    raise client_error(MISSING[kind], op=method)
                if method == "get_parameter":
                    return {"Parameter": {"Value": w.ssm_values.get(kw[key], "")}}
                if method == "get_function_url_config":
                    return {"FunctionUrl": URL}
                if method == "describe_table":
                    return {"Table": {"TableStatus": "DELETING" if (self.region, kind, kw[key]) in w.deleting else "ACTIVE"}}
                return {}
            if method == "create_function_url_config":
                w.res.add((self.region, "url", kw["FunctionName"]))
                return {"FunctionUrl": URL}
            if method == "invoke":
                ev = json.loads(kw["Payload"])
                path = ev.get("rawPath", "/health")
                req_body = json.loads(ev.get("body") or "{}")
                live = req_body.get("mode") == "live"
                body = {"ok": True, "models": ["M2", "M3b"], "items": 33392, "live_configured": True} if path == "/health" else {"items": [{"position": i} for i in range(1, 21)], "timings_ms": {"server_total": 12.3, "search": 40.0},
                                                               "diagnostics": {"cold_start": True, "searches": 2 if live else 0,
                                                                               "fallback": w.live_fallback if live else None}}
                return {"StatusCode": 200, "Payload": io.BytesIO(json.dumps({"statusCode": 200, "body": json.dumps(body)}).encode())}
            if method == "get_function_url_config":
                if (self.region, "url", kw["FunctionName"]) not in w.res:
                    raise client_error("ResourceNotFoundException", op=method)
                return {"FunctionUrl": URL}
            return {}
        return call
