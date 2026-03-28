# Monograph Processed Data Archive

This directory contains plot-ready and monograph-ready processed tables derived
from the current SM89 paper assets and runtime summaries.

Files:

- `inventory_numeric.csv`
  - normalized inventory and frontier counts
- `inventory_plot.csv`
  - compact plot-safe inventory counts for PGFPlots
- `p2r_frontier_numeric.csv`
  - encoded `P2R` frontier status table with simple state scores
- `p2r_frontier_plot.csv`
  - compact plot-safe `P2R` state scores for PGFPlots
- `uplop3_runtime_class_counts.csv`
  - inert vs stable-but-different class counts
- `uplop3_runtime_class_plot.csv`
  - compact plot-safe runtime class counts
- `uplop3_runtime_sites.csv`
  - site-level runtime-class listing
- `uplop3_live_site_numeric.csv`
  - live-site rank, role, jaccard, and distance-to-1
- `uplop3_live_site_plot.csv`
  - compact plot-safe live-site jaccard scores
- `uplop3_pair_baseline_numeric.csv`
  - pair-baseline same/diff counts and normalized diff ratios
- `uplop3_pair_baseline_plot.csv`
  - compact plot-safe pair-baseline divergence ratios
- `tool_effectiveness_numeric.csv`
  - normalized tool-role priorities for semantic workflow plots
- `SHA256SUMS`
  - checksum manifest for processed tables, split figure files, and the built PDF

Primary consumers:

- [MONOGRAPH_SM89_SYNTHESIS.md](../../MONOGRAPH_SM89_SYNTHESIS.md)
- [sm89_monograph.tex](../../tex/sm89_monograph.tex)
- [SHA256SUMS](SHA256SUMS)
