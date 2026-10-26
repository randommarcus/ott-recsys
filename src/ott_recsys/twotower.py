"""Two-tower candidate generation in TensorFlow/Keras (offline only).

Item tower  : [ID embedding | projected centered-SBERT vector] -> dense -> L2-normalise (64-d)
User tower  : mean of item-tower outputs of the user's last N liked items -> dense -> L2-normalise
Loss        : in-batch sampled softmax with logQ correction (Yi et al., 2019)
Serving     : TwoTowerRetrieval is pure numpy (item vectors + one dense layer), no TensorFlow.

THIS MODULE MUST NEVER TOUCH THE TEST PORTION (tests/test_module2.py scans it)."""
import time

import numpy as np
import pandas as pd

VARIANTS = ("id", "content", "both")


# ---------------------------------------------------------------- data: histories and pairs
def user_histories(fit_df: pd.DataFrame, catalog: np.ndarray, n: int, users=None):
    """Each user's last n LIKED items (catalog indices, oldest->newest, left-aligned,
    -1 padded), using everything in fit_df. Returns (user_ids, hist[U, n])."""
    idx = pd.Index(catalog)
    lk = fit_df[fit_df["label"] == 1]
    if users is not None:
        lk = lk[lk["user_id"].isin(set(users))]
    lk = lk.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    uid, it = lk["user_id"].to_numpy(), idx.get_indexer(lk["item_id"])
    if len(uid) == 0:
        return np.empty(0, dtype=uid.dtype), np.empty((0, n), dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(uid)) + 1]
    ends = np.r_[starts[1:], len(uid)]
    hist = np.full((len(starts), n), -1, dtype=np.int32)
    for k, (s, e) in enumerate(zip(starts, ends)):
        h = it[max(s, e - n):e]
        hist[k, :len(h)] = h
    return uid[starts], hist


def build_pairs(train_df: pd.DataFrame, catalog: np.ndarray, n: int, min_history: int = 1) -> dict:
    """Training pairs (history -> next liked item). LEAKAGE RULE: for a liked item at time t the
    history holds only the user's liked items with timestamp STRICTLY LESS than t (the newest n of
    them). Ties in timestamp therefore never see each other, and nothing from the future can enter."""
    idx = pd.Index(catalog)
    lk = train_df[train_df["label"] == 1].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    uid, ts, it = lk["user_id"].to_numpy(), lk["timestamp"].to_numpy(), idx.get_indexer(lk["item_id"])
    assert (it >= 0).all(), "train item missing from catalog"
    starts = np.r_[0, np.flatnonzero(np.diff(uid)) + 1]
    ends = np.r_[starts[1:], len(uid)]
    ar = np.arange(n)[None, :]
    out = {"user": [], "pos": [], "hist": [], "ts": []}
    for s, e in zip(starts, ends):
        t, items = ts[s:e], it[s:e]
        left = np.searchsorted(t, t, side="left")  # number of likes strictly earlier than each one
        lo = np.maximum(0, left - n)
        lens = left - lo
        keep = lens >= min_history
        if not keep.any():
            continue
        pos_idx = lo[keep][:, None] + ar
        valid = ar < lens[keep][:, None]
        out["hist"].append(np.where(valid, items[np.minimum(pos_idx, len(items) - 1)], -1).astype(np.int32))
        out["pos"].append(items[keep].astype(np.int32))
        out["user"].append(np.full(keep.sum(), uid[s]))
        out["ts"].append(t[keep])
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}


# ---------------------------------------------------------------- loss
def sampled_softmax_loss(u, v, pos_ids, log_q, tau: float, use_logq: bool = True):
    """In-batch sampled softmax. Row i's positive is column i; the other rows' positives are its
    negatives. logQ correction: subtract each item's log sampling probability from its logit, so
    popular items are not over-punished just for appearing in many batches. Accidental hits (the
    same item is another row's positive too) are masked out."""
    import tensorflow as tf
    logits = tf.matmul(u, v, transpose_b=True) / tau
    if use_logq:
        logits -= tf.gather(log_q, pos_ids)[None, :]
    b = tf.shape(u)[0]
    same = tf.equal(pos_ids[:, None], pos_ids[None, :]) & tf.logical_not(tf.eye(b, dtype=tf.bool))
    logits = tf.where(same, tf.constant(-1e9, logits.dtype), logits)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.range(b), logits))


def log_sampling_prob(pos: np.ndarray, n_items: int) -> np.ndarray:
    """log Q_j: probability that a random training pair has item j as its positive."""
    counts = np.bincount(pos, minlength=n_items).astype(np.float64)
    return np.log(np.maximum(counts, 1.0) / counts.sum()).astype(np.float32)


# ---------------------------------------------------------------- model
def _build_model_class():
    import tensorflow as tf

    class TwoTower(tf.keras.Model):
        def __init__(self, n_items, content, variant, dim=64, id_dropout=0.25):
            super().__init__()
            assert variant in VARIANTS
            self.n_items, self.variant, self.dim = n_items, variant, dim
            self.use_id, self.use_content = variant in ("id", "both"), variant in ("content", "both")
            self.id_dropout = id_dropout if variant == "both" else 0.0  # dropout only helps when content can compensate
            if self.use_id:
                self.id_emb = tf.keras.layers.Embedding(n_items + 1, dim)  # row n_items = "unknown item"
            if self.use_content:
                assert content is not None and len(content) == n_items
                self.content = tf.constant(content, dtype=tf.float32)
                self.proj = tf.keras.layers.Dense(dim)
            self.mix = tf.keras.layers.Dense(dim)
            self.user_dense = tf.keras.layers.Dense(dim)

        def item_table(self, training: bool, known=None):
            """[n_items, dim] unit-length item vectors (whole catalog in one pass)."""
            parts = []
            if self.use_id:
                ids = tf.range(self.n_items)
                if training and self.id_dropout > 0:
                    ids = tf.where(tf.random.uniform([self.n_items]) < self.id_dropout, self.n_items, ids)
                elif not training and known is not None:
                    ids = tf.where(known, ids, self.n_items)  # items never seen in training use the unknown vector
                parts.append(self.id_emb(ids))
            if self.use_content:
                parts.append(self.proj(self.content))
            x = parts[0] if len(parts) == 1 else tf.concat(parts, axis=1)
            return tf.math.l2_normalize(self.mix(x), axis=1)

        def user_vec(self, table, hist):
            """Mean of the history items' vectors (padding ignored) -> dense -> unit length."""
            mask = tf.cast(hist >= 0, tf.float32)
            g = tf.gather(table, tf.maximum(hist, 0)) * mask[..., None]
            pooled = tf.reduce_sum(g, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)
            return tf.math.l2_normalize(self.user_dense(pooled), axis=1)

    return TwoTower


def make_model(n_items, content, variant, dim=64, id_dropout=0.25):
    m = _build_model_class()(n_items, content, variant, dim, id_dropout)
    m.item_table(False)  # build all weights
    m.user_vec(m.item_table(False), np.zeros((1, 1), dtype=np.int32))
    return m


def export_inference(model, known_mask: np.ndarray) -> dict:
    """Everything serving needs: unit item vectors (unknown-ID items already handled) and the
    user-tower dense layer."""
    import tensorflow as tf
    table = model.item_table(False, known=tf.constant(known_mask)).numpy().astype(np.float32)
    W, b = model.user_dense.get_weights()
    return {"item_vecs": table, "W": W.astype(np.float32), "b": b.astype(np.float32)}


def better(a: dict, b: dict | None, tol: float = 1e-4) -> bool:
    """Model selection: higher Recall@200; if within tol, higher Recall@50 breaks the tie."""
    if b is None:
        return True
    if abs(a["recall@200"] - b["recall@200"]) < tol:
        return a["recall@50"] > b["recall@50"]
    return a["recall@200"] > b["recall@200"]


def train(pairs: dict, n_items: int, content, variant: str, use_logq: bool = True, dim: int = 64,
          tau: float = 0.07, batch_size: int = 2048, lr: float = 1e-3, id_dropout: float = 0.25,
          max_epochs: int = 15, patience: int = 2, fixed_epochs: int | None = None, eval_fn=None,
          known_mask=None, seed: int = 0, log=print):
    """Train; with eval_fn, early-stop/select on Recall@200 (tie-break Recall@50). With
    fixed_epochs, train exactly that many epochs and never evaluate. Returns (model, history)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = make_model(n_items, content, variant, dim, id_dropout)
    opt = tf.keras.optimizers.Adam(lr)
    log_q = tf.constant(log_sampling_prob(pairs["pos"], n_items))
    hist_all, pos_all = pairs["hist"], pairs["pos"]
    steps = len(pos_all) // batch_size
    rng = np.random.default_rng(seed)

    @tf.function(reduce_retracing=True)
    def step(h, p):
        with tf.GradientTape() as tape:
            table = model.item_table(True)
            u = model.user_vec(table, h)
            loss = sampled_softmax_loss(u, tf.gather(table, p), p, log_q, tau, use_logq)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history, best, best_w, bad = [], None, None, 0
    for epoch in range(1, (fixed_epochs or max_epochs) + 1):
        t0 = time.time()
        perm = rng.permutation(len(pos_all))
        losses = [float(step(hist_all[perm[i * batch_size:(i + 1) * batch_size]],
                             pos_all[perm[i * batch_size:(i + 1) * batch_size]])) for i in range(steps)]
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "train_s": time.time() - t0}
        if eval_fn is not None and fixed_epochs is None:
            rec.update(eval_fn(model))
            if better(rec, best):
                best, best_w, bad = rec, model.get_weights(), 0
            else:
                bad += 1
        history.append(rec)
        log("epoch " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if eval_fn is not None and fixed_epochs is None and bad >= patience:
            log(f"early stop: no improvement for {patience} epochs")
            break
    if best_w is not None:
        model.set_weights(best_w)
    return model, history, (best or {}).get("epoch", len(history))


# ---------------------------------------------------------------- retrieval wrapper (numpy only)
class TwoTowerRetrieval:
    """Harness-compatible model built from exported arrays. Users with no liked items in the fitting
    data fall back to popularity (counts of likes)."""
    name = "two_tower"

    def __init__(self, item_vecs, W, b, history_len: int, users=None):
        self.V, self.W, self.b, self.n, self.users = item_vecs, W, b, history_len, users

    def fit(self, fit_df, catalog):
        uid, hist = user_histories(fit_df, catalog, self.n, self.users)
        pop = np.bincount(pd.Index(catalog).get_indexer(fit_df.loc[fit_df["label"] == 1, "item_id"]),
                          minlength=len(catalog)).astype(np.float32)
        self.prepare(uid, hist, pop)

    def prepare(self, uid, hist, pop):
        """Compute user vectors from ready-made histories (lets training reuse them every epoch)."""
        mask = (hist >= 0).astype(np.float32)
        pooled = (self.V[np.maximum(hist, 0)] * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1.0)
        z = pooled @ self.W + self.b
        self.U = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        self.u_index = {u: i for i, u in enumerate(uid)}
        self.pop = pop / max(pop.max(), 1.0)

    def no_likes(self, users) -> np.ndarray:
        return np.array([u not in self.u_index for u in users])

    def score(self, user_ids, query_hours=None):
        rows = np.array([self.u_index.get(u, -1) for u in user_ids])
        s = self.U[np.maximum(rows, 0)] @ self.V.T
        s[rows < 0] = self.pop[None, :]  # popularity fallback
        return s
