# SM89 Paper Section Coverage

This file maps the current paper claims and paper assets to the existing paper
outline sections.

## Section 4. Results: ISA Comparison

Currently supportable now:

- Table A1: SM89 inventory summary
- bounded Ada-only inventory/result paragraph from:
  - `C01`
  - `C02`

Still blocked on future evidence:

- Pascal-vs-Ada unique instruction delta tables
- full cross-architecture instruction-count comparison

## Abstract and Section 10. Conclusion

Currently supportable now:

- bounded Ada-only summary of:
  - `C01`-`C03`
  - `C04`-`C17`
- explicit language that the broader Pascal-vs-Ada comparison remains scaffolded
  but not fully evidenced yet
- standalone manuscript fragment:
  [PAPER_DRAFT_SM89.md](PAPER_DRAFT_SM89.md)

Still blocked on future evidence:

- strong cross-architecture headline claims about latency, throughput, and ISA
  evolution across Pascal and Ada

## Section 7. Results: Binary Encoding

Currently supportable now:

- Table A2: `P2R` frontier status
- Table A3: `UPLOP3` structural boundary
- bounded Ada-only frontier prose from:
  - `C04`-`C11`

Still blocked on future evidence:

- full direct Pascal-vs-Ada encoding field comparison
- cross-architecture control-word comparison

## Section 8. Discussion

Currently supportable now:

- Figure A2: `UPLOP3` runtime class map
- Table A4: live `UPLOP3` site ranking
- Table A5: tool effectiveness matrix
- bounded synthesis prose from:
  - `C12`-`C18`

Still blocked on future evidence:

- strong cross-architecture causal claims about ISA evolution that depend on
  the missing Pascal-side measurements

## Section 9. Reproducibility

Currently supportable now:

- claim ledger:
  [PAPER_CLAIMS_MATRIX.md](PAPER_CLAIMS_MATRIX.md)
- coverage plan:
  [PAPER_FIGURE_TABLE_PLAN.md](PAPER_FIGURE_TABLE_PLAN.md)
- instantiated assets:
  [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- manuscript-fragment verifier:
  [verify_paper_draft.py](scripts/verify_paper_draft.py)

## Bounded-Language Rule

For the current paper draft:

- do say:
  - local cubin-side `P2R.B1/B2/B3` materialization is proven
  - direct local source/IR `P2R.B*` emission is not yet reproduced
  - local cubin-side `PLOP3 -> UPLOP3` is structurally valid
  - direct source/IR `UPLOP3.LUT` emission is not yet reproduced
- do not say:
  - source-level `P2R.B*` has been reproduced
  - source-level `UPLOP3.LUT` has been reproduced
