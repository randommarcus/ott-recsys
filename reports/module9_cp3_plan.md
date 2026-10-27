# Module 9, checkpoint 3 plan: loading the Module 4b exports into Zilliz (PLAN ONLY, nothing written)

Status: proposal, 2026-10-27. Only read-only research and local file reads were done for this plan; Zilliz has not been contacted and `.env` was not read.
Targets: the free cluster in aws-eu-central-1. Every write below waits for your explicit go.

## 0. Findings that change earlier assumptions (please read first)
1. **Index type.** The Module 4b schema files say HNSW (M=16, efConstruction=200). Zilliz's own docs/FAQ say Zilliz Cloud clusters support only **AUTOINDEX** (their managed index; metric IP is allowed), and an older support article mentions HNSW/DiskANN, so the docs conflict.
   Whether the FREE tier is restricted further is **not confirmed**. Plan: create with AUTOINDEX + IP, no `ef`. If AUTOINDEX is refused, stop and report (no silent fallback to another index).
2. **The "read-only key" may not be able to search.** Zilliz documents the built-in cluster role "Read-Only" as view-only (describe/list; no vector operations listed), and cluster users/roles exist only for **Dedicated** clusters, not Free. On a Free cluster the least-privilege tool is a *customized API key* with a project role. The
   project role **Data Viewer** has Read (not Write) on cluster data, but the docs I could reach do not say in so many words that it can Search/Query or that Insert is refused. The verification script (section 6) settles both facts empirically. If Data Viewer cannot search, the decision is yours:
   (a) use a Read-Write key in SSM (it could modify data), or (b) serve in precomputed mode only and keep live retrieval as a local demonstration.
3. **My Lambda REST client has two likely mistakes.** Zilliz's REST v2 search page spells the parameter `searchParams: {"metricType": "IP"}`; my `serving/search.py` sends `metric_type` plus `params.ef`. For AUTOINDEX the `ef` should go away. I will fix the client and its test in a separate commit BEFORE the load, then check the real shape in verification (section 6, step A2).
4. Free-plan facts (from the earlier docs read): 5 collections, 5 GB, 4 vector fields per collection, nq <= 10, topK <= 1,024, insert 2 MB/s, auto-suspend after 7 days idle. Cost of this checkpoint: **$0** (free tier). Nothing paid is involved.

## 1. Collections (3 of the 5 allowed)
| Collection | Purpose | Rows | Source files |
|---|---|---|---|
| `movie_items_twotower_id` | main-lane retrieval (warm items only) | **16,450** | `two_tower/export_id_only/items.parquet` (+ `vector.npy`, `item_ids.npy`) |
| `movie_items_content` | new-release lane (filter `is_cold == true`) | **33,392** (16,942 cold, 16,450 warm) | `content/export_content/items.parquet` (+ `vec384.npy`, `vec64.npy`) |
| `ott_write_probe` | scratch target for the "write must be refused" test | 1 | created by the loader |

## 2. Schemas (from the committed `milvus_schema.json` files, unchanged)
- Both items collections: `item_id` INT64 primary key (no auto id); `title` VARCHAR(256) (longest 191); `year` INT32; `genres` VARCHAR(128) (longest 77); `n_train_likes` INT32; `is_cold` BOOL.
- `movie_items_twotower_id`: `vector` FLOAT_VECTOR(64). `movie_items_content`: `vec384` FLOAT_VECTOR(384) (centred SBERT, unit length) and `vec64` FLOAT_VECTOR(64) (PCA). Vectors are unit length, so inner product = cosine.
- `ott_write_probe`: `item_id` INT64 key + `vector` FLOAT_VECTOR(2); one row with id 1.

## 3. Index parameters
AUTOINDEX, metric **IP**, on each of the 4 vector fields (`vector`, `vec384`, `vec64`, probe `vector`). No scalar indexes (33k rows; the `is_cold` filter is cheap). Search params: metric IP only (no `ef`; AUTOINDEX tunes itself). Collections are loaded after indexing.
Consequence: search becomes approximate (the numpy stand-in used so far is exact), so the live-vs-precomputed agreement measured earlier (main lists identical for 600 of 600 users) may drop slightly; section 6 step C measures it.

## 4. Storage against the free tier
- Raw vectors: 4.2 MB (two-tower) + 51.3 MB (vec384) + 8.5 MB (vec64) = **about 64 MB**; strings and ints add a few MB; total about **70 MB against 5 GB (about 1.4%)**. In the docs' own unit (5 GB = 1 million 768-d vectors) we are about 21,000 768-d-equivalents (about 2%).
- **Not confirmed:** whether index files or replicas count toward the 5 GB; the exact meaning of "capacity" on the Free plan (pricing page not reachable). With 70 MB of data a 10x overhead would still fit.
- Insert rate cap 2 MB/s means at least about 35 s; plan on 2-5 minutes with batches of 500 rows (about 1 MB each) and retry with back-off if throttled.

## 5. Load order (new `scripts/load_zilliz.py`; uses ZILLIZ_URI + the full-access ZILLIZ_TOKEN from `.env` inside Python; never printed)
Dry run by default; the real run needs `--confirm LOAD-ZILLIZ` (same pattern as the AWS teardown).
1. **Local integrity check, no network:** dims match the schema; vectors unit length; `item_ids.npy` equals the parquet `item_id` column; parquet vectors equal the `.npy` files; counts as in section 1.
2. **Read-only preflight:** connect, server version, list collections. Refuse if any of our three names exists (unless `--recreate`, which drops ONLY those three names and nothing else).
3. Create + load `movie_items_twotower_id`, then `movie_items_content`, then `ott_write_probe` (smallest, most important first).
   For each: create schema and AUTOINDEX, insert in batches of 500 ordered by `item_id`, flush, load, then confirm the row count equals the expected number.
4. Prints only: step names, row counts, elapsed seconds, error TYPE names. Re-running after a partial failure uses `--recreate` (so no half-filled collection is ever trusted).
Also written: `scripts/zilliz_teardown.py` (dry run by default, `--confirm DROP-ZILLIZ`, drops only these three names) for the end of the project.

## 6. Verification (new `scripts/verify_zilliz.py`; READ-ONLY KEY ONLY)
Reads `ZILLIZ_URI` and `ZILLIZ_TOKEN_READONLY` from `.env` inside Python. It **refuses to run** if the read-only value is blank or equal to `ZILLIZ_TOKEN`, so it cannot silently use the full key. Output limited to counts, yes/no and exception type names; library stdout/stderr swallowed; no traceback; same safeguards and tests as `check_zilliz.py`.
- **A1 search through pymilvus:** (i) two-tower collection, query = the stored vector of one known item, limit 10 -> prints `results: 10` and whether the top hit is the item itself; (ii) content collection `vec384`, filter `is_cold == true`, limit 30 -> `results: 30, all cold: yes`; (iii) `vec64`, limit 5.
- **A2 search through the exact REST path Lambda uses** (`serving/search.py::ZillizREST`): the same queries; prints counts and how many of the top 10 agree with A1. This closes the "REST call shape unverified" risk.
- **B harmless write attempts with the read-only key:** insert one dummy row (id -1, 2-d vector) into the scratch collection `ott_write_probe`, once through pymilvus and once through REST. Expected: both **refused for permission reasons**. The script prints `write refused: yes (pymilvus)` / `(REST)`. Three possible outcomes per attempt:
  `REFUSED` (pass); `ALLOWED (UNEXPECTED)` (fail: the key can write; the only damage is one junk row in the scratch collection; the script tells you to change the key's role); `INCONCLUSIVE` (the failure was not clearly a permission error, e.g. network; fail, because an unrelated error proves nothing).
  Afterwards a search of the probe must still return exactly its one original row (`probe unchanged: yes`).
- **C optional recall check (`--recall`):** for 100 sampled users, retrieve the main and lane candidates through REST and compare with the precomputed lists (needs the local `users.npz`; prints one overlap percentage per lane). Not required by you; I recommend it because section 3 makes search approximate.
- Exit code 0 only if A1, A2 and B all pass.
Tests (before any real run): fake client for every outcome above (refused / unexpectedly allowed / inconclusive / search count wrong / read-only equal to full key), static no-print scan, canary subprocess test with a fake URI and token, mutation-checked like the earlier ones.

## 7. Order of work and checkpoints
| Step | Who | What | Zilliz state |
|---|---|---|---|
| 0 | me, local | fix the REST client (`metricType`, no `ef`) + test; write `load_zilliz.py`, `verify_zilliz.py`, `zilliz_teardown.py` and their tests; commit each script before editing it; run the local integrity check | none |
| 1 | you, Zilliz console | create a **customized API key**: project role **Data Viewer**, restricted to this one cluster; put it in `.env` as `ZILLIZ_TOKEN_READONLY` (never paste it to me). Confirm `.env` also has `ZILLIZ_URI` and the full-access `ZILLIZ_TOKEN`. | none |
| 2 | you, go | loader **dry run** (read-only calls only: connect, list collections, local checks) | none |
| 3 | you, go | real load with `--confirm LOAD-ZILLIZ`: **FIRST ZILLIZ WRITE** (creates 3 collections, about 50,000 rows, about 2-5 min, $0) | 3 collections |
| 4 | you, go | `verify_zilliz.py` (and optionally `--recall`); paste me the output | probe row may appear only if the key can write |
| 5 | me | record results in PROGRESS.md; then checkpoint 4 (deploy) | none |
Rollback at any time: `zilliz_teardown.py --confirm DROP-ZILLIZ` or delete the collections in the console. Later: the free cluster auto-suspends after 7 days idle, so it must be resumed in the console before the demo, and the first search after a resume will be slow (relevant to the latency study).

## 8. Decisions I need from you
1. AUTOINDEX instead of the exported HNSW settings (docs say it is the only managed option; HNSW only if Zilliz confirms Free supports it).
2. Data Viewer on a customized, cluster-restricted API key as the "read-only" key, with the fallback in section 0.2 if it cannot search.
3. Include the optional recall check.
