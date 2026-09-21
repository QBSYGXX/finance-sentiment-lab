"""Download pinned public inputs, then create disjoint model-input splits."""
from collections import Counter, defaultdict
import base64
import hashlib
import json
import logging
import time
import urllib.request

from sklearn.model_selection import train_test_split
from common import ROOT, SEED, LABELS, MAX_LENGTH, normalize, save_json

DATA_REPO = "supersymmetry-technologies/BBT-FinCUGE-Applications"
CODE_REPO = "649453932/Chinese-Text-Classification-Pytorch"
SOURCES = [
    (DATA_REPO, "29998e5571cc664f3f8eb28cfd6caa054ba10028", "data/raw/train_list.json"),
    (DATA_REPO, "bd5fbb00c1e0c84fc5a133f61e6e5d0eb16b3727", "data/raw/eval_list.json"),
    (DATA_REPO, "17116c216ae52aa40925139be748005e1d0a68a2", "data/raw/UPSTREAM_README.md"),
    (CODE_REPO, "7e8cc5f96f92541ceefa6aa63dfafe0d1c6263f3", "third_party/TextCNN.original.py"),
    (CODE_REPO, "434bb7651a77791c223eee550b94201e334b4af6", "third_party/LICENSE.TextCNN"),
]


# 模块：开源来源获取；核心业务：否。
# Git blob 哈希固定输入版本；缓存文件仍须通过哈希核验。
def fetch_blob(repo, sha, relative):
    path = ROOT / relative
    if path.exists():
        data = path.read_bytes()
    else:
        url = f"https://api.github.com/repos/{repo}/git/blobs/{sha}"
        logging.info("Downloading %s", relative)
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "finance-sentiment-lab"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    item = json.load(response)
                data = base64.b64decode(item["content"])
                break
            except Exception:
                logging.exception("Download attempt %s failed", attempt + 1)
                if attempt == 2:
                    raise
                time.sleep(1)
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != sha:
        raise ValueError(f"Source hash mismatch: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"repository": repo, "git_blob": sha, "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


# 模块：去重与数据划分；核心业务：是。
# 使用实际模型输入去重；冲突标签不猜测，测试文本优先保留并从训练池删除。
def clean_splits(raw_train, raw_test):
    grouped = defaultdict(list)
    for origin, rows in [("source_train", raw_train), ("source_eval", raw_test)]:
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or row[1] not in (0, 1, 2):
                raise ValueError(f"Unexpected labeled row at {origin}:{index}")
            normalized = normalize(row[0])
            if normalized:
                grouped[normalized].append((origin, index, int(row[1]), row[0]))
    train_pool, test, conflicts = [], [], []
    duplicates = 0
    overlap = 0
    for normalized, members in grouped.items():
        labels = {x[2] for x in members}
        if len(labels) != 1:
            conflicts.append({"text": normalized, "labels": sorted(labels), "rows": len(members)})
            continue
        heldout = [x for x in members if x[0] == "source_eval"]
        origin, index, label, original = (heldout or members)[0]
        overlap += int(bool(heldout) and any(x[0] == "source_train" for x in members))
        duplicates += len(members) - 1
        item = {"id": hashlib.sha256(normalized.encode()).hexdigest()[:16],
                "text": normalized, "label": label, "source": origin,
                "source_row": index, "truncated": len(original) > MAX_LENGTH}
        (test if heldout else train_pool).append(item)
    train, val = train_test_split(train_pool, test_size=0.15, random_state=SEED,
                                 stratify=[r["label"] for r in train_pool])
    splits = {"train": train, "validation": val, "test": test}
    sets = {k: {r["text"] for r in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    audit = {"raw_train": len(raw_train), "raw_eval": len(raw_test),
             "duplicate_rows_removed": duplicates, "cross_source_overlaps": overlap,
             "conflicting_texts_removed": len(conflicts),
             "conflicting_rows_removed": sum(x["rows"] for x in conflicts),
             "split_counts": {k: len(v) for k, v in splits.items()},
             "class_counts": {k: {LABELS[i]: sum(r["label"] == i for r in v) for i in range(3)} for k, v in splits.items()},
             "split_intersections": 0, "max_characters": MAX_LENGTH, "seed": SEED,
             "protocol": "Source train -> stratified train/validation; source eval -> held-out test. Deduplicated by normalized truncated input. Not the official leaderboard protocol.",
             "domain": "Financial social-media comments; no separate headline/body or timestamps."}
    return splits, audit, conflicts


def prepare():
    sources = [fetch_blob(*row) for row in SOURCES]
    raw_train = json.loads((ROOT / "data/raw/train_list.json").read_text(encoding="utf-8"))
    raw_test = json.loads((ROOT / "data/raw/eval_list.json").read_text(encoding="utf-8"))
    splits, audit, conflicts = clean_splits(raw_train, raw_test)
    for name, rows in splits.items():
        save_json(ROOT / "data" / f"{name}.json", rows)
    save_json(ROOT / "data/conflicts.json", conflicts)
    save_json(ROOT / "data/audit.json", audit)
    save_json(ROOT / "data/sources.json", sources)
    logging.info("Prepared data: %s", json.dumps(audit, ensure_ascii=False))

