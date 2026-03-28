# SM89 Monograph Build Note

This note captures the reproducible build path for the current monograph-level
SM89 synthesis.

## Inputs

Core source documents:

- [MONOGRAPH_SM89_SYNTHESIS.md](MONOGRAPH_SM89_SYNTHESIS.md)
- [MONOGRAPH_GLOSSARY.md](MONOGRAPH_GLOSSARY.md)
- [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)

Processed numeric and plot inputs:

- [processed/monograph_20260323/README.md](processed/monograph_20260323/README.md)
- [SHA256SUMS](processed/monograph_20260323/SHA256SUMS)

TeX package:

- [sm89_monograph.tex](tex/sm89_monograph.tex)
- [appendix_selection_laws.tex](tex/appendix_selection_laws.tex)
- [fig_inventory_closure.tex](tex/fig_inventory_closure.tex)
- [fig_p2r_frontier.tex](tex/fig_p2r_frontier.tex)
- [fig_uplop3_runtime.tex](tex/fig_uplop3_runtime.tex)
- [fig_pair_baseline.tex](tex/fig_pair_baseline.tex)

## Reproducible Build

From the repository root:

```sh
src/sass_re/scripts/build_monograph_pdf.sh
```

This performs, in order:

1. regenerate the processed monograph archive,
2. compile the TeX package twice with `pdflatex`,
3. verify PDF freshness against TeX and plot inputs,
4. write the checksum manifest,
5. re-run the monograph asset verifier.

## Verification Gates

```sh
python3 src/sass_re/scripts/verify_monograph_assets.py
python3 src/sass_re/scripts/verify_monograph_pdf.py
python3 src/sass_re/scripts/verify_paper_assets.py
python3 src/sass_re/scripts/verify_paper_draft.py
```

## Primary Output

- [sm89_monograph.pdf](tex/build/sm89_monograph.pdf)

## Boundary

This build note is Ada-only and SM89-only. It does not claim that the larger
Pascal-vs-Ada paper is fully populated or fully buildable from the current
evidence base.
