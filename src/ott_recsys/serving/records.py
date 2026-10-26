"""The per-user record stored in DynamoDB (table ott-users). Arrays travel as raw little-endian bytes and counters as ints, so no floats
ever reach DynamoDB (its Python client rejects floats). One record is a few KB.

Fields (all optional except user_id and last_ts):
  cand        int32[230]  precomputed candidate list (catalog indices, -1 = empty slot): 200 main + 30 new-release lane   [mode A]
  ttvec       float32[64] two-tower user vector (absent = user has no liked history -> popularity fallback)
  ncf_ug/um   float32[32] NCF user vectors after the offline fold-in (absent = NCF does not know the user)
  prof        float32[384] unit-length mean of the centred text embeddings of everything the user liked (content profile)
  share       float32[20] share of the user's liked items in each genre
  seen        int32[n]    catalog indices of everything the user ever rated (liked or not): never recommended again
  last_ts     int         timestamp of the user's last recorded activity (the simulated clock starts here)
  sess_len, sess_likes    length and number of likes of the rating spree that ends at last_ts (gaps <= 30 min)
  sess_items  int32[<=50] the last items of that spree (oldest first)"""
import numpy as np

ARRAYS = {"cand": np.int32, "ttvec": np.float32, "ncf_ug": np.float32, "ncf_um": np.float32, "prof": np.float32,
          "share": np.float32, "seen": np.int32, "sess_items": np.int32}
INTS = ("user_id", "last_ts", "sess_len", "sess_likes", "ncf_known")


def pack(rec: dict) -> dict:
    out = {k: int(rec[k]) for k in INTS if rec.get(k) is not None}
    for k, dt in ARRAYS.items():
        v = rec.get(k)
        if v is not None and len(v):
            out[k] = np.ascontiguousarray(v, dtype=np.dtype(dt).newbyteorder("<")).tobytes()
    return out


def unpack(item: dict) -> dict:
    rec = {k: int(item[k]) for k in INTS if k in item}  # DynamoDB numbers arrive as Decimal
    for k, dt in ARRAYS.items():
        if k in item:
            raw = item[k]
            raw = getattr(raw, "value", raw)  # boto3 wraps binary in Binary(.value)
            rec[k] = np.frombuffer(bytes(raw), dtype=np.dtype(dt).newbyteorder("<")).astype(dt)
        else:
            rec[k] = None
    return rec
