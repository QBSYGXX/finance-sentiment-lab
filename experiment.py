"""Fixed protocol: validation selects models; held-out evaluation runs afterwards."""
import csv
from datetime import datetime, timezone
import hashlib
import logging
import random
import re
import time

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, SEED, LABELS, MAX_LENGTH, load_json, save_json
from model import TextCNN, build_vocab, encode


# 模块：统一评估指标；核心业务：是。
# Macro-F1 对三类等权，避免多数类掩盖少数类表现。
def metrics(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
            "per_class": classification_report(y, predicted, labels=[0, 1, 2], target_names=LABELS,
                                                output_dict=True, zero_division=0)}


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


@torch.inference_mode()
def cnn_probabilities(model, tokens, device):
    model.eval()
    chunks = []
    for batch in tokens.split(128):
        chunks.append(model(batch.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(chunks)


# 模块：传统模型基线；核心业务：是。
# TF-IDF 只在训练集拟合；验证集选 C，测试集不参与选择。
def train_linear(train, validation):
    best_model, best_score, trials = None, -1, []
    for c in (1.0, 4.0):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                                    max_features=40000, sublinear_tf=True)),
            ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=600, random_state=SEED)),
        ])
        logging.info("Training TF-IDF + logistic regression, C=%s", c)
        started = time.perf_counter()
        pipeline.fit([r["text"] for r in train], [r["label"] for r in train])
        result = metrics([r["label"] for r in validation], pipeline.predict_proba([r["text"] for r in validation]))
        trials.append({"C": c, "validation_macro_f1": result["macro_f1"], "seconds": time.perf_counter() - started})
        logging.info("Linear C=%s validation macro-F1=%.4f", c, result["macro_f1"])
        if result["macro_f1"] > best_score:
            best_model, best_score = pipeline, result["macro_f1"]
    joblib.dump(best_model, ROOT / "artifacts/linear.joblib")
    return best_model, trials


# 模块：TextCNN 对照实验；核心业务：是。
# 只改变损失函数类别权重，保持种子、模型、批次顺序及学习率相同。
def train_cnn(name, weighted, train, validation, vocab, epochs, device):
    seed_all()
    started = time.perf_counter()
    model = TextCNN(len(vocab) + 2).to(device)
    x_train, x_val = encode([r["text"] for r in train], vocab), encode([r["text"] for r in validation], vocab)
    y_train = torch.tensor([r["label"] for r in train], dtype=torch.long)
    y_val = np.array([r["label"] for r in validation])
    weights = None
    if weighted:
        counts = torch.bincount(y_train, minlength=3).float()
        weights = (len(y_train) / (3 * counts)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    history, best_f1, best_epoch, stale = [], -1, 0, 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    logging.info("%s parameters=%s batches/epoch=%s device=%s", name, sum(p.numel() for p in model.parameters()), len(loader), device)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for step, (tokens, labels) in enumerate(loader, start=1):
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            if step % 50 == 0:
                logging.info("%s epoch=%s/%s step=%s/%s loss=%.4f", name, epoch, epochs, step, len(loader), loss.item())
        result = metrics(y_val, cnn_probabilities(model, x_val, device))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train),
                        "validation_macro_f1": result["macro_f1"], "validation_accuracy": result["accuracy"]})
        logging.info("%s epoch=%s validation macro-F1=%.4f accuracy=%.4f", name, epoch, result["macro_f1"], result["accuracy"])
        if result["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = result["macro_f1"], epoch, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ROOT / "artifacts" / f"{name}.pt")
        else:
            stale += 1
            if stale >= 2:
                logging.info("%s early stop after two non-improving epochs", name)
                break
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith("cuda") else 0
    model.load_state_dict(torch.load(ROOT / "artifacts" / f"{name}.pt", map_location=device, weights_only=True))
    return model, {"history": history, "best_epoch": best_epoch, "validation_macro_f1": best_f1,
                   "seconds": time.perf_counter() - started, "peak_allocated_vram_mib": peak,
                   "parameters": sum(p.numel() for p in model.parameters()), "class_weighted": weighted}


# 模块：图表与误判记录；核心业务：否。
def export_results(report, test, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = ROOT / "artifacts/figures"
    figures.mkdir(exist_ok=True)
    names = list(report["models"])
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3.5), constrained_layout=True)
    for ax, name in zip(axes, names):
        matrix = np.array(report["models"][name]["test"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Greens")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() * .55 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["Neg", "Neu", "Pos"],
               yticklabels=["Neg", "Neu", "Pos"], xlabel="Predicted", ylabel="True", title=name)
    fig.savefig(figures / "confusion.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.3), constrained_layout=True)
    for name in ("cnn_plain", "cnn_weighted"):
        history = report["models"][name]["training"]["history"]
        ax.plot([x["epoch"] for x in history], [x["validation_macro_f1"] for x in history], marker="o", label=name)
    ax.set(xlabel="Epoch", ylabel="Validation macro-F1")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figures / "learning.png", dpi=150)
    plt.close(fig)
    with (ROOT / "artifacts/predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["id", "text", "true_label"] + [f"{n}_{suffix}" for n in names for suffix in ["prediction", "score"]]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(test):
            record = {"id": row["id"], "text": row["text"], "true_label": LABELS[row["label"]]}
            for name in names:
                record[f"{name}_prediction"] = LABELS[int(predictions[name][i].argmax())]
                record[f"{name}_score"] = round(float(predictions[name][i].max()), 4)
            writer.writerow(record)


def train_all(epochs, device):
    if epochs < 1:
        raise ValueError("Epoch count must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    splits = {name: load_json(ROOT / "data" / f"{name}.json") for name in ("train", "validation", "test")}
    train, validation, test = splits.values()
    train_inputs = {r["text"] for r in train}
    val_inputs = {r["text"] for r in validation}
    test_inputs = {r["text"] for r in test}
    assert not (train_inputs & val_inputs or train_inputs & test_inputs or val_inputs & test_inputs)
    seed_all()
    logging.info("Training started; torch=%s GPU=%s", torch.__version__, torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU")
    started = time.perf_counter()
    linear, trials = train_linear(train, validation)
    vocab = build_vocab([r["text"] for r in train])
    save_json(ROOT / "artifacts/vocab.json", vocab)
    models, training = {}, {}
    for name, weighted in [("cnn_plain", False), ("cnn_weighted", True)]:
        models[name], training[name] = train_cnn(name, weighted, train, validation, vocab, epochs, device)
    selected_cnn = max(training, key=lambda k: training[k]["validation_macro_f1"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "labels": LABELS,
              "audit": load_json(ROOT / "data/audit.json"), "models": {}, "selected_cnn": selected_cnn,
              "selection_rule": "Validation macro-F1 only; held-out test evaluated after training and selection.",
              "data_hashes": {k: hashlib.sha256((ROOT / "data" / f"{k}.json").read_bytes()).hexdigest() for k in splits},
              "environment": {"torch": torch.__version__, "device": device,
                              "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None},
              "limitations": ["Single seed exploratory study", "Financial social comments, not news or stock-return labels",
                              "Conflict removal changes the evaluation population", "Model scores are not calibrated probabilities"]}
    test_y = np.array([r["label"] for r in test])
    predictions = {"linear": linear.predict_proba([r["text"] for r in test])}
    report["models"]["linear"] = {"training": {"trials": trials},
        "validation_macro_f1": max(x["validation_macro_f1"] for x in trials),
        "test": metrics(test_y, predictions["linear"])}
    for name, model in models.items():
        predictions[name] = cnn_probabilities(model, encode([r["text"] for r in test], vocab), device)
        report["models"][name] = {"training": training[name], "validation_macro_f1": training[name]["validation_macro_f1"],
                                   "test": metrics(test_y, predictions[name])}
    majority = int(np.bincount([r["label"] for r in train]).argmax())
    report["majority_baseline"] = metrics(test_y, np.tile(np.eye(3)[majority], (len(test), 1)))
    report["seconds"] = time.perf_counter() - started
    chosen = max(report["models"], key=lambda k: report["models"][k]["validation_macro_f1"])
    report["selected_model"] = chosen
    negation = np.array([bool(re.search(r"不|没|未|无|别", r["text"])) for r in test])
    report["negation_slice"] = {"count": int(negation.sum()), "definition": "Contains 不/没/未/无/别; exploratory diagnostic only",
                               "models": {n: metrics(test_y[negation], p[negation]) for n, p in predictions.items()}}
    errors = []
    for i, row in enumerate(test):
        if any(int(p[i].argmax()) != row["label"] for p in predictions.values()):
            errors.append({"id": row["id"], "text": row["text"], "label": LABELS[row["label"]],
                           "models": {n: {"label": LABELS[int(p[i].argmax())], "score": float(p[i].max())} for n, p in predictions.items()}})
    save_json(ROOT / "artifacts/errors.json", errors)
    export_results(report, test, predictions)
    save_json(ROOT / "artifacts/report.json", report)
    lines = ["# 首轮实验结果", "", "数据为 FinFE 金融社交文本，使用去重后的自定义划分，不与官方榜单直接比较。", "",
             "| 模型 | 验证集 Macro-F1 | 测试准确率 | 测试 Macro-F1 |", "|---|---:|---:|---:|"]
    for name, item in report["models"].items():
        lines.append(f"| {name} | {item['validation_macro_f1']:.4f} | {item['test']['accuracy']:.4f} | {item['test']['macro_f1']:.4f} |")
    lines += ["", f"验证集选中的模型：{chosen}；CNN 配置：{selected_cnn}。", "", "测试集仅在模型选择后用于报告结果。",
              "", "## 如何理解", "", "Macro-F1 分别计算三种情绪的 F1 后取平均，每类权重相同。准确率则受多数类占比影响。",
              "", "类别加权仅提高少数类错误的训练惩罚，不保证最终得分更高；请依据本次结果判断。",
              "", "## 边界", "", "本次为单随机种子探索实验，未验证财经新闻迁移效果或交易价值。所有模型均使用最多128字输入。",
              "没有标题字段，因此没有开展标题与正文消融。TextCNN 为随机初始化的轻量神经网络，不是预训练模型。",
              "", "图表：figures/confusion.png、figures/learning.png。逐条结果：predictions.csv。"]
    (ROOT / "artifacts/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    for name, item in report["models"].items():
        logging.info("FINAL %s test accuracy=%.4f macro-F1=%.4f", name, item["test"]["accuracy"], item["test"]["macro_f1"])
    logging.info("Experiment finished in %.1fs; selected using validation=%s", report["seconds"], chosen)
