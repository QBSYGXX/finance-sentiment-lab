"""Recompute exported metrics and verify the saved input/model contract."""
import csv
import hashlib
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from common import ROOT, LABELS, load_json, save_json


# 模块：产物验收；核心业务：否。
def main():
    report = load_json(ROOT / 'artifacts/report.json')
    with (ROOT / 'artifacts/predictions.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == report['audit']['split_counts']['test']
    for split, expected in report['data_hashes'].items():
        assert hashlib.sha256((ROOT / 'data' / f'{split}.json').read_bytes()).hexdigest() == expected
    y = [LABELS.index(r['true_label']) for r in rows]
    for name, item in report['models'].items():
        pred = [LABELS.index(r[f'{name}_prediction']) for r in rows]
        assert abs(f1_score(y, pred, average='macro') - item['test']['macro_f1']) < 1e-10
        assert abs(accuracy_score(y, pred) - item['test']['accuracy']) < 1e-10
        assert sum(map(sum, item['test']['confusion_matrix'])) == len(rows)
    assert report['selected_model'] == max(report['models'], key=lambda n: report['models'][n]['validation_macro_f1'])
    for name in ('confusion.png', 'learning.png'):
        image = np.array(Image.open(ROOT / 'artifacts/figures' / name).convert('RGB'))
        assert image.std() > 5, f'Blank figure: {name}'
    hashes = {name: hashlib.sha256((ROOT / 'artifacts' / name).read_bytes()).hexdigest()
              for name in ('linear.joblib', 'cnn_plain.pt', 'cnn_weighted.pt', 'vocab.json')}
    save_json(ROOT / 'artifacts/verification.json', {'passed': True, 'test_rows': len(rows), 'model_sha256': hashes})
    print('PASS: exported metrics, split hashes, selected model, figures, and model files')


if __name__ == '__main__':
    main()
