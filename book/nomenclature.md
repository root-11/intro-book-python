# Nomenclature

Quick reference for symbols, notation, and abbreviations the book uses. Concept *definitions* live in the [glossary](../concepts/glossary.md); this page covers the shorthand only.

## Symbols

| Symbol | Meaning |
|---|---|
| §N | Section number — e.g., §5 refers to section 5. |
| → | Leads to / becomes / transitions to. Appears in section titles (e.g., §29 "10K → 1M") and prose. |
| `[!NOTE]` / `[!TIP]` / `[!WARNING]` | Callout box — content the reader should pay particular attention to. |

## Text formatting

| Form | Meaning |
|---|---|
| `monospace` | Code: types, variable names, function names, file paths. |
| *italic* | First definition of a term, or emphasis. |
| **bold** | A term being highlighted as load-bearing in the current paragraph. |
| `# anti-pattern: bad!` | A code comment that flags the snippet as something the chapter is arguing *against*. The label travels with the code if a reader copy-pastes. |

## Variables you will see across chapters

| Variable | Meaning |
|---|---|
| `i`, `j` | Index into a column. `i` is the index of the row currently under discussion. |
| `t` or `tick` | Tick number — the simulator's step counter. |
| `id` | Stable entity identifier (a small unsigned integer; usually `np.uint32`). |
| `gen` | Generation counter, paired with a slot index to detect stale references (§10). |
| `pos_x`, `pos_y` | Position columns of a creature (`np.float32`). |
| `vel_x`, `vel_y` | Velocity columns of a creature (`np.float32`). |
| `to_remove`, `to_insert` | Buffers of pending mutations applied at end-of-tick (§22). |
| `n_active` | Length of the live prefix of a fixed-capacity column (§21, §24). |

## Python types and their numpy counterparts

This book uses numpy's typed dtypes for hot data. The mapping the reader will see most often:

| Python | numpy | size | range |
|---|---|---|---|
| `int` (CPython, ≤ 2³⁰) | — | 28 bytes | unbounded |
| — | `np.int8`  | 1 byte | -128 to 127 |
| — | `np.uint8` | 1 byte | 0 to 255 |
| — | `np.int32` | 4 bytes | ±2¹ |
| — | `np.uint32` | 4 bytes | 0 to 2³² |
| — | `np.int64` | 8 bytes | ±2⁶³ |
| `float` (CPython) | `np.float64` | 8 bytes (CPython has 24-byte object overhead) | ~15 decimal digits |
| — | `np.float32` | 4 bytes | ~7 decimal digits |
| — | `np.bool_` | 1 byte (in arrays) | True / False |

## Conventions for code blocks

| Form | Convention |
|---|---|
| Plain triple-backtick `python` | A snippet to read; not necessarily complete. |
| Snippet with `# anti-pattern: bad!` first line | A snippet shown as the *wrong* way; the chapter is about the right way. |
| `uv run code/measurement/<file>.py` | A measurement exhibit the reader can run on their machine. The numbers in the chapter were measured the same way. |
