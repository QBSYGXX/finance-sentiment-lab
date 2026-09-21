"""Shared paths and text contract."""
from pathlib import Path
import json
import logging
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent
LABELS = ["消极", "中性", "积极"]
MAX_LENGTH = 128
SEED = 42


# 模块：文本输入口径；核心业务：是。
# 训练、测试与页面使用同一套规范化及截断规则。
def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()[:MAX_LENGTH]


# 模块：文件与日志；核心业务：否。
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def setup_log(name):
    (ROOT / "logs").mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / f"{name}.log", encoding="utf-8"),
    ], force=True)

