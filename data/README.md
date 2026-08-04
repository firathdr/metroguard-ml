# Data

MetroGuard uses the **MetroPT-3** dataset from the UCI Machine Learning Repository:

- DOI: https://doi.org/10.24432/C5VW3R
- Dataset page: https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Creators: Narjes Davari, Bruno Veloso, Rita Ribeiro, and João Gama

The raw 208 MB file is intentionally excluded from Git. Run:

```bash
metroguard data download
metroguard data prepare
```

The published CSV contains 1,516,948 rows from one Air Production Unit. Although the UCI
metadata contains both 1 Hz and 0.1 Hz statements, the timestamps in the released file are
approximately 10 seconds apart with jitter and longer gaps. MetroGuard therefore calculates
coverage from timestamps instead of assuming a perfectly regular signal.

`data/demo/replay.csv` is a compact attributed extract used only for the historical research
demo. It remains subject to CC BY 4.0. Raw and processed directories are gitignored.

