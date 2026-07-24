# Fair-transport (ICML 2023)

This directory vendors the demographic-parity post-processor from:

- Ruicheng Xian, Lang Yin, and Han Zhao, *Fair and Optimal Classification via
  Post-Processing*, ICML 2023.
- Upstream: https://github.com/uiuctml/fair-classification
- Tag: `icml.23`
- Commit: `ff83c13c3c17de95ac7a29c0889727665014a08a`
- Upstream file: `postprocess.py`

Local changes are limited to validation, explicit solver selection/fallback,
and solver diagnostics. The Wasserstein-barycenter LP and score-map extraction
follow the archived upstream implementation.
