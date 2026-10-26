"""Zilliz connection settings for the Lambda function, read from ONE SSM SecureString parameter (/ott/zilliz = JSON with uri and token).

The token is a READ-ONLY key. Nothing here ever prints, logs or formats the values: the object hides them from repr/str/format, errors carry only the
name of the exception type, and the original exception is dropped (`from None`) so its message cannot leak through a traceback."""
import json

PARAM_NAME = "/ott/zilliz"


class ConfigError(Exception):
    """Message holds an exception TYPE name only."""


class ZillizConfig:
    __slots__ = ("_uri", "_token")

    def __init__(self, uri: str, token: str):
        self._uri, self._token = str(uri), str(token)

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def token(self) -> str:
        return self._token

    def __repr__(self) -> str:
        return "ZillizConfig(<hidden>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "ZillizConfig(<hidden>)"

    def __reduce__(self):
        raise TypeError("ZillizConfig cannot be pickled")


def load_from_ssm(name: str = PARAM_NAME, client=None) -> ZillizConfig:
    try:
        if client is None:
            import boto3
            client = boto3.client("ssm")
        raw = client.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
        d = json.loads(raw)
        return ZillizConfig(d["uri"], d["token"])
    except Exception as exc:
        raise ConfigError(f"could not load the Zilliz settings: {type(exc).__name__}") from None
