"""Local-only inference and evaluation browser, using the Python standard library."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import threading
from urllib.parse import urlparse

import joblib
import torch
from common import ROOT, LABELS, MAX_LENGTH, load_json, normalize
from model import TextCNN, encode


# 模块：已训练模型推理；核心业务：是。
# 页面推理使用 CPU，保留 GPU 给后续实验；加载本项目生成的权重。
class Predictor:
    def __init__(self):
        torch.set_num_threads(2)
        self.linear = joblib.load(ROOT / "artifacts/linear.joblib")
        self.vocab = load_json(ROOT / "artifacts/vocab.json")
        self.models = {}
        self.lock = threading.Lock()
        for name in ("cnn_plain", "cnn_weighted"):
            model = TextCNN(len(self.vocab) + 2)
            model.load_state_dict(torch.load(ROOT / "artifacts" / f"{name}.pt", map_location="cpu", weights_only=True))
            self.models[name] = model.eval()

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入非空文本")
        if len(text) > 2000:
            raise ValueError("文本不能超过2000字")
        normalized = normalize(text)
        with self.lock, torch.inference_mode():
            probabilities = {"linear": self.linear.predict_proba([normalized])[0]}
            tokens = encode([normalized], self.vocab)
            for name, model in self.models.items():
                probabilities[name] = model(tokens).softmax(1)[0].numpy()
        return {"input": normalized, "truncated": len(text) > MAX_LENGTH,
                "models": {name: {"label": LABELS[int(p.argmax())], "score": float(p.max()),
                                    "scores": [float(x) for x in p], "review": bool(p.max() < .6)} for name, p in probabilities.items()}}


# 模块：本地 HTTP 接口；核心业务：否。
# 固定文件白名单，避免通过 URL 读取本机其他文件。
def serve(port):
    predictor = Predictor()
    report = load_json(ROOT / "artifacts/report.json")
    errors = load_json(ROOT / "artifacts/errors.json")
    routes = {
        "/": ROOT / "static/index.html",
        "/style.css": ROOT / "static/style.css",
        "/app.js": ROOT / "static/app.js",
        "/confusion.png": ROOT / "artifacts/figures/confusion.png",
        "/learning.png": ROOT / "artifacts/figures/learning.png",
        "/predictions.csv": ROOT / "artifacts/predictions.csv",
        "/icons/play.svg": ROOT / "static/icons/play.svg",
        "/icons/rotate-ccw.svg": ROOT / "static/icons/rotate-ccw.svg",
        "/icons/download.svg": ROOT / "static/icons/download.svg",
    }

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            content = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/report":
                return self.send_json(200, report)
            if path == "/api/errors":
                return self.send_json(200, errors)
            if path == "/health":
                return self.send_json(200, {"status": "ready", "models": 3, "application": "finance_sentiment_lab"})
            if path not in routes or not routes[path].exists():
                return self.send_json(404, {"error": "Not found"})
            file = routes[path]
            content = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(file.name)[0] or "application/octet-stream") + ("; charset=utf-8" if file.suffix in (".html", ".css", ".js", ".csv") else ""))
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/api/predict":
                return self.send_json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32000:
                    return self.send_json(413, {"error": "请求大小无效"})
                if "application/json" not in self.headers.get("Content-Type", ""):
                    return self.send_json(415, {"error": "需要JSON请求"})
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求格式无效")
                result = predictor.predict(payload.get("text"))
                self.send_json(200, result)
            except (ValueError, TypeError):
                self.send_json(400, {"error": "请输入1至2000字的文本"})
            except Exception:
                logging.exception("Inference failed")
                self.send_json(500, {"error": "模型分析失败，请查看本地日志"})

        def log_message(self, format, *args):
            logging.info("HTTP " + format, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    logging.info("READY http://127.0.0.1:%s", port)
    server.serve_forever()
