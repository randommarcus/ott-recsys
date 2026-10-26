"""Data access for the handler: DynamoDB in AWS, an in-memory twin for tests and local runs.

ott-users  key user_id (N): the user record (see records.py). Read-only for the function.
ott-bandit key arm_id (N): counters n (impressions) and s (SIMULATED likes) for one new-release arm. arm_id = catalog index.
           The function reads with BatchGetItem and writes with UpdateItem ADD (atomic counters)."""
import numpy as np

from .records import pack, unpack


class MemoryStore:
    """Behaves like the DynamoDB tables: stores the PACKED form, so a bad field type fails here too."""

    def __init__(self):
        self._users, self._arms = {}, {}
        self.calls = {"get_user": 0, "get_arms": 0, "add_impressions": 0, "add_feedback": 0}

    def put_user(self, rec: dict) -> None:
        self._users[int(rec["user_id"])] = pack(rec)

    def get_user(self, user_id: int):
        self.calls["get_user"] += 1
        item = self._users.get(int(user_id))
        return None if item is None else unpack(item)

    def get_arms(self, arm_ids):
        self.calls["get_arms"] += 1
        a = [self._arms.get(int(i), (0, 0)) for i in arm_ids]
        return np.array([x[0] for x in a], float), np.array([x[1] for x in a], float)

    def add_impressions(self, arm_ids) -> None:
        self.calls["add_impressions"] += 1
        for i in arm_ids:
            n, s = self._arms.get(int(i), (0, 0))
            self._arms[int(i)] = (n + 1, s)

    def add_feedback(self, arm_id: int, reward: int) -> None:
        self.calls["add_feedback"] += 1
        n, s = self._arms.get(int(arm_id), (0, 0))
        self._arms[int(arm_id)] = (n, s + int(reward))


class DynamoStore:
    def __init__(self, users_table: str = "ott-users", bandit_table: str = "ott-bandit", resource=None):
        if resource is None:
            import boto3
            resource = boto3.resource("dynamodb")
        self._db, self._users_name, self._bandit_name = resource, users_table, bandit_table
        self._users, self._bandit = resource.Table(users_table), resource.Table(bandit_table)

    def get_user(self, user_id: int):
        item = self._users.get_item(Key={"user_id": int(user_id)}).get("Item")
        return None if item is None else unpack(item)

    def get_arms(self, arm_ids):
        ids = [int(i) for i in arm_ids]
        found = {}
        for a in range(0, len(ids), 100):  # BatchGetItem allows 100 keys
            keys = [{"arm_id": i} for i in ids[a:a + 100]]
            req = {self._bandit_name: {"Keys": keys, "ProjectionExpression": "arm_id, n, s"}}
            for _ in range(5):
                resp = self._db.batch_get_item(RequestItems=req)
                for it in resp["Responses"].get(self._bandit_name, []):
                    found[int(it["arm_id"])] = (int(it.get("n", 0)), int(it.get("s", 0)))
                req = resp.get("UnprocessedKeys") or {}
                if not req:
                    break
        return np.array([found.get(i, (0, 0))[0] for i in ids], float), np.array([found.get(i, (0, 0))[1] for i in ids], float)

    def add_impressions(self, arm_ids) -> None:
        for i in arm_ids:
            self._bandit.update_item(Key={"arm_id": int(i)}, UpdateExpression="ADD n :one", ExpressionAttributeValues={":one": 1})

    def add_feedback(self, arm_id: int, reward: int) -> None:
        self._bandit.update_item(Key={"arm_id": int(arm_id)}, UpdateExpression="ADD s :r", ExpressionAttributeValues={":r": int(reward)})
