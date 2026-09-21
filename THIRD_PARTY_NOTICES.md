# Third-party notices

## TextCNN

- Repository: https://github.com/649453932/Chinese-Text-Classification-Pytorch
- Copyright (c) 2019 huwenxing. MIT License.
- Original code: `third_party/TextCNN.original.py`.
- License: `third_party/LICENSE.TextCNN`.
- `model.py` adapts this structure with smaller dimensions, tensor input,
  padding-window masking and a vocabulary fitted only on training inputs.
- Pinned Git blobs are recorded in `prepare.py` and `docs/results/sources.json`.

## Lucide

- Repository: https://github.com/lucide-icons/lucide
- Version: lucide-static 0.468.0.
- Bundled assets: `static/icons/play.svg`, `rotate-ccw.svg`, `download.svg`.
- Upstream license is retained verbatim in `third_party/LICENSE.Lucide`.

## FinFE data

- Repository: https://github.com/supersymmetry-technologies/BBT-FinCUGE-Applications
- Path: `FinCUGE_Publish/finfe`.
- No clear independent data license was identified during project preparation.
- No dataset texts, per-example predictions, error records, vocabulary or trained
  weights are redistributed here. `prepare.py` retrieves pinned inputs from the
  upstream repository for local use. Users must check the upstream terms.
- Aggregate evaluation metrics, counts, source hashes and charts are included
  under `docs/`. The project MIT license does not license the upstream data.
