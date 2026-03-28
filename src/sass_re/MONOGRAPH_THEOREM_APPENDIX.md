# SM89 Frontier Theorem Appendix

This appendix isolates the tightest paper-safe theorem-like statements that the
current repository evidence supports.

## Appendix A. `P2R` Selection Law

### Definition

For a target family `T = P2R.B*`, define:

- `F_T(s) = 1` iff direct local source/IR lowering emits `T`
- `G_T(b) = 1` iff cubin-side substitution materializes runnable `T`

### Proposition A1

On the current SM89 stack, `G_T = 1` for `P2R.B1`, `P2R.B2`, and `P2R.B3`,
while the observed direct-local source/IR search gives `F_T = 0`.

### Corollary A1

The current unresolved `P2R.B*` gap is not an opcode-existence gap.

### Bounded interpretation

The repository evidence supports the narrower claim that the gap is a
source/IR-level form-selection problem on the tested toolchains and frontends.
It does not prove impossibility for all future sysroots, compilers, or hidden
IR layers.

## Appendix B. `UPLOP3` Structural Law

### Definition

Let the source patch classes be:

- `U`: `ULOP3 -> UPLOP3`
- `P`: `PLOP3 -> UPLOP3`

### Proposition B1

`U` is structurally invalid in the tested local cubin contexts, while `P` is
structurally valid in multiple local contexts.

### Corollary B1

The local substrate for `UPLOP3` materialization is predicate-logic form
(`PLOP3`), not ordinary uniform integer logic form (`ULOP3`).

## Appendix C. `UPLOP3` Runtime-Class Law

### Definition

For a patched local context `k`, define:

- inert: no tested pattern family changes the observed output
- stable-but-different: at least one tested pattern family changes the observed
  output while the kernel remains runnable and sanitizer-clean

### Proposition C1

The tested local `UPLOP3` contexts partition into inert and stable-but-
different classes.

### Proposition C2

The strongest currently ranked live local anchors are `uniform_occ1` and
`cutlass_occ5`; `uniform_occ2` is a secondary anchor; `cutlass_occ4` acts most
strongly as a widener/amplifier.

### Bounded interpretation

This is a classification result over the tested local contexts and pattern
families, not a complete global taxonomy over all possible `UPLOP3` contexts.
