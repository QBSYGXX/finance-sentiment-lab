"""End-to-end local API checks against the running trained application."""
import json
import sys
from pathlib import Path
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, LABELS, load_json, save_json

url = load_json(ROOT / '.runtime.json')['url']
checks = []
for path in ('/', '/style.css', '/app.js', '/api/report', '/api/errors', '/confusion.png', '/learning.png', '/icons/play.svg', '/icons/rotate-ccw.svg', '/icons/download.svg'):
    with urllib.request.urlopen(url + path, timeout=10) as response:
        assert response.status == 200 and len(response.read()) > 0
    checks.append('GET ' + path)

examples = []
for text in ('业绩增长不错，继续看好后市。', '补仓补得心力憔悴，还是一直跌。', '今天先观望，等公告出来再决定。', '涨' * 200):
    request = urllib.request.Request(url + '/api/predict', data=json.dumps({'text': text}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert len(result['input']) <= 128
    assert set(result['models']) == {'linear', 'cnn_plain', 'cnn_weighted'}
    for model in result['models'].values():
        assert len(model['scores']) == 3 and abs(sum(model['scores']) - 1) < 1e-5
        assert model['label'] == LABELS[max(range(3), key=lambda i: model['scores'][i])]
    examples.append({'text': text[:40], 'models': {k: v['label'] for k, v in result['models'].items()}})
checks.append('Valid prediction, probability normalization, and truncation')

for payload in ({'text': ''}, {'text': 123}, {'text': '涨' * 2001}, []):
    request = urllib.request.Request(url + '/api/predict', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(request, timeout=10)
        raise AssertionError('Invalid request was accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
checks.append('Invalid input rejected')
save_json(ROOT / 'artifacts/http_verification.json', {'passed': True, 'checks': checks, 'examples': examples,
          'browser_visual_check': 'Not completed: Chrome extension unavailable; no IAB fallback used.'})
print(json.dumps({'passed': True, 'checks': len(checks), 'examples': examples}, ensure_ascii=False, indent=2))
