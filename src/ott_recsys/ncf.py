"""Neural Collaborative Filtering (NeuMF, He et al. 2017) as a RANKER feature.

NCF scores (user, item) pairs from ID embeddings; it is used on candidate lists, never on the whole
catalog. Users and items with no train likes have no trained embedding, so the scorer returns NaN
and a 'known' flag for them (it never invents a score).

Training uses train positives only. This module never touches validation or test data
(tests/test_module2.py scans it)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

MLP_LAYERS = (64, 32, 16)


# ---------------------------------------------------------------- training data
def build_training_data(train_df: pd.DataFrame) -> dict:
    """Positives = train likes. 'Known' users/items = those with >= 1 train like (only they get embeddings)."""
    likes = train_df[train_df["label"] == 1]
    users, items = np.sort(likes["user_id"].unique()), np.sort(likes["item_id"].unique())
    pos_u = pd.Index(users).get_indexer(likes["user_id"]).astype(np.int32)
    pos_i = pd.Index(items).get_indexer(likes["item_id"]).astype(np.int32)
    return {"users": users, "items": items, "pos_u": pos_u, "pos_i": pos_i,
            "item_counts": np.bincount(pos_i, minlength=len(items))}


def sample_negatives(pos_u, pos_i, n_items: int, k: int, pop_frac: float, item_counts, rng, max_tries: int = 5):
    """k negatives per positive: round(k*pop_frac) drawn in proportion to count**0.75, the rest uniform over
    known items. A user's own train likes are never drawn (collisions are redrawn; any left are dropped)."""
    n = len(pos_u)
    n_pop = int(round(k * pop_frac))
    cdf = np.cumsum(item_counts.astype(np.float64) ** 0.75)
    cdf /= cdf[-1]
    is_pop_col = np.arange(k) < n_pop

    def draw(shape_cols_pop):
        return np.minimum(np.searchsorted(cdf, rng.random(shape_cols_pop)), n_items - 1)

    items = np.empty((n, k), dtype=np.int64)
    if n_pop:
        items[:, :n_pop] = draw((n, n_pop))
    if k > n_pop:
        items[:, n_pop:] = rng.integers(0, n_items, (n, k - n_pop))
    users = np.repeat(pos_u.astype(np.int64), k).reshape(n, k)
    pos_keys = np.unique(pos_u.astype(np.int64) * n_items + pos_i)

    def collides(u, i):
        key = u * n_items + i
        j = np.searchsorted(pos_keys, key)
        return (j < len(pos_keys)) & (pos_keys[np.minimum(j, len(pos_keys) - 1)] == key)

    bad = collides(users, items)
    for _ in range(max_tries):
        if not bad.any():
            break
        r, c = np.nonzero(bad)
        pop_cell = is_pop_col[c]
        new = np.where(pop_cell, draw(len(r)), rng.integers(0, n_items, len(r)))
        items[r, c] = new
        bad[r, c] = collides(users[r, c], new)
    keep = ~bad
    return users[keep].astype(np.int32), items[keep].astype(np.int32)


# ---------------------------------------------------------------- numpy-only scorer
class NCFScorer:
    """Pure numpy forward pass of NeuMF. score() returns logits, NaN where the user or item is unknown."""

    def __init__(self, arrays: dict, user_ids, item_ids):
        self.a = arrays
        self.user_ids, self.item_ids = np.asarray(user_ids), np.asarray(item_ids)
        self._u, self._i = pd.Index(self.user_ids), pd.Index(self.item_ids)

    def known(self, user_ids, item_ids):
        return self._u.get_indexer(user_ids) >= 0, self._i.get_indexer(item_ids) >= 0

    def score(self, user_ids, item_ids, chunk: int = 200_000) -> np.ndarray:
        ui, ii = self._u.get_indexer(user_ids), self._i.get_indexer(item_ids)
        out = np.full(len(ui), np.nan, dtype=np.float32)
        ok = np.flatnonzero((ui >= 0) & (ii >= 0))
        a = self.a
        for s in range(0, len(ok), chunk):
            sel = ok[s:s + chunk]
            u, i = ui[sel], ii[sel]
            g = a["ug"][u] * a["ig"][i]
            h = np.concatenate([a["um"][u], a["im"][i]], axis=1)
            for l in range(len(MLP_LAYERS)):
                h = np.maximum(h @ a[f"W{l}"] + a[f"b{l}"], 0.0)
            out[sel] = (np.concatenate([g, h], axis=1) @ a["Wo"] + a["bo"])[:, 0]
        return out

    def shuffled_users(self, seed: int = 0) -> "NCFScorer":
        """Same model with user embeddings reassigned among users (personalisation check)."""
        perm = np.random.default_rng(seed).permutation(len(self.user_ids))
        a = dict(self.a)
        a["ug"], a["um"] = self.a["ug"][perm], self.a["um"][perm]
        return NCFScorer(a, self.user_ids, self.item_ids)

    def save(self, path: Path) -> None:
        np.savez(path, user_ids=self.user_ids, item_ids=self.item_ids, **self.a)

    @classmethod
    def load(cls, path: Path) -> "NCFScorer":
        z = np.load(path)
        return cls({k: z[k] for k in z.files if k not in ("user_ids", "item_ids")}, z["user_ids"], z["item_ids"])


def score_candidates(scorer: NCFScorer, users, cand, catalog) -> np.ndarray:
    """[U, W] NCF logits for candidate lists (catalog indices, -1 = padding); NaN = cannot score / padding."""
    cand = np.asarray(cand)
    flat_items = np.where(cand >= 0, np.asarray(catalog)[np.maximum(cand, 0)], -1).ravel()
    flat_users = np.repeat(np.asarray(users), cand.shape[1])
    s = scorer.score(flat_users, flat_items).reshape(cand.shape)
    s[cand < 0] = np.nan
    return s


# ---------------------------------------------------------------- TensorFlow model
def _build_model_class():
    import tensorflow as tf
    L = tf.keras.layers

    class NeuMF(tf.keras.Model):
        def __init__(self, n_users, n_items, gmf=32, mlp=32, init_std=0.05):
            super().__init__()
            init = tf.keras.initializers.RandomNormal(0.0, init_std)
            self.ug, self.ig = L.Embedding(n_users, gmf, embeddings_initializer=init), L.Embedding(n_items, gmf, embeddings_initializer=init)
            self.um, self.im = L.Embedding(n_users, mlp, embeddings_initializer=init), L.Embedding(n_items, mlp, embeddings_initializer=init)
            self.mlp = [L.Dense(h, activation="relu") for h in MLP_LAYERS]
            self.out = L.Dense(1)

        def call(self, u, i):
            g = self.ug(u) * self.ig(i)
            h = tf.concat([self.um(u), self.im(i)], axis=1)
            for layer in self.mlp:
                h = layer(h)
            return tf.squeeze(self.out(tf.concat([g, h], axis=1)), axis=1)

    return NeuMF


def make_model(n_users: int, n_items: int):
    import tensorflow as tf
    m = _build_model_class()(n_users, n_items)
    m(tf.zeros([1], tf.int32), tf.zeros([1], tf.int32))  # build
    return m


def model_arrays(model) -> dict:
    a = {"ug": model.ug.get_weights()[0], "ig": model.ig.get_weights()[0],
         "um": model.um.get_weights()[0], "im": model.im.get_weights()[0]}
    for l, layer in enumerate(model.mlp):
        a[f"W{l}"], a[f"b{l}"] = layer.get_weights()
    a["Wo"], a["bo"] = model.out.get_weights()
    return {k: v.astype(np.float32) for k, v in a.items()}


def better(a: dict, b: dict | None, tol: float = 1e-4) -> bool:
    """Selection: higher NDCG@10 over the candidate lists; within tol, higher Recall@20 wins."""
    if b is None:
        return True
    if abs(a["ndcg@10"] - b["ndcg@10"]) < tol:
        return a["recall@20"] > b["recall@20"]
    return a["ndcg@10"] > b["ndcg@10"]


def train(data: dict, neg_k: int = 4, pop_frac: float = 0.5, batch: int = 4096, lr: float = 1e-3, max_epochs: int = 30,
          patience: int = 3, fixed_epochs: int | None = None, eval_fn=None, seed: int = 0, log=print):
    """Train NeuMF with BCE and fresh sampled negatives every epoch. With eval_fn, early-stop on it
    (eval_fn(scorer) -> {'ndcg@10','recall@20'}); with fixed_epochs, train exactly that many and never evaluate."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    nu, ni = len(data["users"]), len(data["items"])
    model = make_model(nu, ni)
    opt = tf.keras.optimizers.Adam(lr)
    rng = np.random.default_rng(seed)

    @tf.function(reduce_retracing=True)
    def step(u, i, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model(u, i)))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history, best, best_w, bad = [], None, None, 0
    for epoch in range(1, (fixed_epochs or max_epochs) + 1):
        t0 = time.time()
        nu_, ni_ = sample_negatives(data["pos_u"], data["pos_i"], ni, neg_k, pop_frac, data["item_counts"], rng)
        u = np.concatenate([data["pos_u"], nu_])
        i = np.concatenate([data["pos_i"], ni_])
        y = np.concatenate([np.ones(len(data["pos_u"]), np.float32), np.zeros(len(nu_), np.float32)])
        perm = rng.permutation(len(u))
        losses = [float(step(u[perm[s:s + batch]], i[perm[s:s + batch]], y[perm[s:s + batch]]))
                  for s in range(0, len(u), batch)]
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "rows": int(len(u)), "train_s": time.time() - t0}
        if eval_fn is not None and fixed_epochs is None:
            rec.update(eval_fn(NCFScorer(model_arrays(model), data["users"], data["items"])))
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
    return NCFScorer(model_arrays(model), data["users"], data["items"]), history, (best or {}).get("epoch", len(history))


def fold_in_users(frozen: "NCFScorer", pos_user_ids, pos_item_ids, neg_k: int = 4, pop_frac: float = 0.5, epochs: int = 3,
                  lr: float = 1e-3, batch: int = 4096, seed: int = 0, log=print):
    """FOLD-IN: re-optimise ONLY the user embeddings on the given positives, with every item embedding and the whole MLP frozen
    (asserted at the end). Used before the test run so that likes made during validation reach NCF without retraining it.
    Positives on items the frozen model has no embedding for are dropped (NCF cannot score them). Users the frozen model has no
    embedding for get a seeded random start. Returns a new NCFScorer over the folded-in users and the per-epoch history."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    a = frozen.a
    ii_all = frozen._i.get_indexer(np.asarray(pos_item_ids))
    keep = ii_all >= 0
    pu_raw, ii = np.asarray(pos_user_ids)[keep], ii_all[keep].astype(np.int32)
    users_new = np.sort(np.unique(pu_raw))
    ui = pd.Index(users_new).get_indexer(pu_raw).astype(np.int32)
    nu, ni = len(users_new), len(frozen.item_ids)
    rng = np.random.default_rng(seed)
    ug0 = rng.normal(0.0, 0.05, (nu, a["ug"].shape[1])).astype(np.float32)
    um0 = rng.normal(0.0, 0.05, (nu, a["um"].shape[1])).astype(np.float32)
    old = frozen._u.get_indexer(users_new)
    ug0[old >= 0], um0[old >= 0] = a["ug"][old[old >= 0]], a["um"][old[old >= 0]]  # known users start from their trained vectors
    model = make_model(nu, ni)
    model.ug.set_weights([ug0]); model.um.set_weights([um0])
    model.ig.set_weights([a["ig"]]); model.im.set_weights([a["im"]])
    for l, layer in enumerate(model.mlp):
        layer.set_weights([a[f"W{l}"], a[f"b{l}"]])
    model.out.set_weights([a["Wo"], a["bo"]])
    train_vars = [model.ug.trainable_variables[0], model.um.trainable_variables[0]]  # ONLY these are updated
    opt = tf.keras.optimizers.Adam(lr)
    counts = np.bincount(ii, minlength=ni)

    @tf.function(reduce_retracing=True)
    def step(u, i, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model(u, i)))
        opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
        return loss

    history = []
    for epoch in range(1, epochs + 1):
        nu_, ni_ = sample_negatives(ui, ii, ni, neg_k, pop_frac, counts, rng)
        u, i = np.concatenate([ui, nu_]), np.concatenate([ii, ni_])
        y = np.concatenate([np.ones(len(ui), np.float32), np.zeros(len(nu_), np.float32)])
        perm = rng.permutation(len(u))
        loss = float(np.mean([float(step(u[perm[s:s + batch]], i[perm[s:s + batch]], y[perm[s:s + batch]])) for s in range(0, len(u), batch)]))
        history.append({"epoch": epoch, "loss": loss, "rows": int(len(u))})
        log(f"fold-in epoch {epoch}: loss {loss:.4f}, {len(u)} rows")
    new = model_arrays(model)
    for k in ("ig", "im", "Wo", "bo", *[f"{p}{l}" for l in range(len(MLP_LAYERS)) for p in ("W", "b")]):
        assert np.array_equal(new[k], a[k]), f"fold-in changed the frozen parameter {k}"
    return NCFScorer(new, users_new, frozen.item_ids), history
