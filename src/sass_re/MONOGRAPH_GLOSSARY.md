# SM89 Monograph Glossary

## Symbols

- `S`
  - space of local source programs
- `I`
  - space of intermediate representations
- `B`
  - space of cubin-level binaries
- `D`
  - space of disassembled SASS neighborhoods
- `R`
  - space of observable runtime behaviors under a test family
- `P`
  - a chosen semantic pattern family
- `T`
  - a target opcode family, such as `P2R.B*` or `UPLOP3.LUT`
- `C : S -> I`
  - frontend lowering map
- `A : I -> B`
  - assembler/packer map
- `N : B -> D`
  - disassembly neighborhood extraction map
- `E : B x P -> R`
  - runtime execution map under pattern family `P`
- `F_T`
  - direct source/IR frontier indicator
- `G_T`
  - cubin-side validity indicator
- `H_T`
  - semantic deviation indicator

## Terms

- anchor
  - a live `UPLOP3` site that remains the strongest stable semantic reference
    point in a local family
- amplifier
  - a live site that broadens an already-live branch more than it serves as a
    standalone anchor
- sensitizer
  - a live site that changes conditions under which a branch widens, without
    being the strongest standalone anchor
- inert
  - a patched cubin site that decodes and executes but matches baseline on the
    tested semantic patterns
- stable-but-different
  - a patched cubin site that decodes and executes and produces reproducible
    semantic deltas
- form selection
  - the compiler/IR-level choice of one low-level opcode form instead of a
    semantically equivalent neighborhood
- opcode existence
  - the question of whether a local target can materially decode and execute a
    given opcode at all
- pair baseline
  - a local stable two-site patched context used as a reference point for
    analyzing which third site acts as a widener or stabilizer

## Core Claims In Glossary Form

- `P2R.B*`:
  - currently a form-selection frontier, not an opcode-existence frontier
- `UPLOP3.LUT`:
  - currently a structural-plus-semantic cubin-side frontier with ranked live
    sites and modifiers
