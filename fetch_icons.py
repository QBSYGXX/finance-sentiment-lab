"""Fetch the small MIT/ISC-licensed Lucide assets used by the local page."""
import urllib.request
from common import ROOT

target = ROOT / 'static/icons'
target.mkdir(parents=True, exist_ok=True)
for name in ('play', 'rotate-ccw', 'download'):
    url = f'https://cdn.jsdelivr.net/npm/lucide-static@0.468.0/icons/{name}.svg'
    print('Downloading', name, flush=True)
    with urllib.request.urlopen(url, timeout=20) as response:
        content = response.read()
    if b'<svg' not in content:
        raise ValueError('Unexpected icon content')
    (target / f'{name}.svg').write_bytes(content)
print('Icons ready', flush=True)
with urllib.request.urlopen('https://cdn.jsdelivr.net/npm/lucide-static@0.468.0/LICENSE', timeout=20) as response:
    license_text = response.read()
(ROOT / 'third_party/LICENSE.Lucide').write_bytes(license_text)
