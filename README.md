# Rcn

`rcn` — **R**ust **C**ompressor, **N**ew — is a leveled command-line compressor
written in Rust. It combines a hash-chain LZ wire engine, context mixing,
Burrows–Wheeler Transform (BWT), and DP-LZP matching.

Optimization tickets, A/B history, and measured numbers:
[DEVELOPMENT.md](DEVELOPMENT.md) · [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) ·
[BENCHMARKS.md](BENCHMARKS.md).

## Goal

**Primary success metric:** each numbered level beats its peer class on
**both** compression ratio (lower `ratio%`) **and** compress throughput
(higher cmp MB/s) on the headline 5-file set (dickens, webster, nci, mr,
json). A file **both**-wins when ratio is a win or tie **and** cmp speed
is a win. Decode MB/s is reported but is not the v1 gate.
**2026-11-03 update:** PGO build verified on Apple Silicon (stable-aarch64-apple-darwin); binary Indirect model (W5) added to binary/exec stacks; wire chain depth increased to 8 probes (T1); binary DP window 4→8 MiB (T2); BWT trial threshold 1MB→2MB (T3); Order-12 PPMd tested but no improvement over order-8; benchmarks regenerated from .work/bench5 (rcn-1, -3, -9, fast on dickens, json, mr).

| Level | Engine | Must beat on both axes |
|------:|--------|------------------------|
| `-1` | Wire: hash-chain LZ + optional order-0 rANS | `lz4 -9`, `zstd -1` |
| `-3` | General: byte CM + DP-LZP, no BWT trials | `gzip -9` |
| `-9` | Archive (default): hybrid Fast Text + Slow Binary | `zstd -19` (stretch: xz, brotli) |
| `-19` | Max: Slow bit CM on every block | xz / brotli (ratio stretch) |

CLI aliases: `--level 1` / `-1`, `3` / `-3`, `9` / `-9`, `19` / `-19`.
`--mode hybrid|slow|fast|wire|general` remains. Default is **`-9` / hybrid**.

One setting cannot beat `lz4`/`zstd -1` on speed **and** `xz`/`brotli` on
ratio. Peers themselves do not do that. Levels own a peer class each.

## Scorecards

Verdict cells are `ratio / cmp-speed / both`. `both=win` only if ratio is
win or tie **and** cmp speed is win. json near-zero ratios are ties.
Measured numbers: [BENCHMARKS.md](BENCHMARKS.md). Refreshed 2026-11-03
(T1 wire chain, T2 DP window, T3 BWT trial). Regenerate:
`scripts/bench_vs_sota.sh <corpus_dir>`.

**rcn `-1` vs speed peers:**

| file | vs lz4-9 | vs zstd-1 |
|------|:--------:|:---------:|
| dickens | lose / **win** / split | lose / lose / lose |
| webster | lose / lose / lose | lose / lose / lose |
| nci | lose / lose / lose | lose / lose / lose |
| mr | **win** / **win** / **win** | lose / lose / lose |
| json | tie / **win** / **win** | tie / **win** / **win** |

**rcn `-3` vs gzip -9:**

| file | vs gzip-9 |
|------|:---------:|
| dickens | lose / lose / lose |
| webster | lose / **win** / split |
| nci | lose / **win** / split |
| mr | **win** / lose / split |
| json | lose / lose / lose |

**rcn `-9` vs `zstd -19` (default gate):**

| file | vs zstd-19 |
|------|:----------:|
| dickens | **win** / **win** / **win** |
| webster | **win** / **win** / **win** |
| nci | **win** / **win** / **win** |
| mr | **win** / lose / split |
| json | tie / lose / split |

**rcn `-9` vs other peers:**

| file | vs xz-9 | vs brotli-11 | vs gzip-9 | vs lz4-9 | vs zstd-1 |
|------|:-------:|:------------:|:---------:|:--------:|:---------:|
| dickens | **win** / **win** / **win** | **win** / **win** / **win** | **win** / lose / split | **win** / lose / split | **win** / lose / split |
| webster | lose / **win** / split | tie / **win** / **win** | **win** / lose / split | **win** / lose / split | **win** / lose / split |
| nci | **win** / **win** / **win** | lose / **win** / split | **win** / lose / split | **win** / lose / split | **win** / lose / split |
| mr | **win** / lose / split | **win** / lose / split | **win** / lose / split | **win** / lose / split | **win** / lose / split |
| json | tie / lose / split | tie / lose / split | tie / lose / split | tie / lose / split | tie / lose / split |

## Design methodology

`rcn` is a staged pipeline. Cheap levels skip expensive stages; archive
levels keep them:

- **Classify first.** Input is split into variable-size blocks by data type so
  each block gets the transform it deserves (see [The method](#the-method)).
  Level `-1` uses large independent chunks and the wire LZ engine.
- **Transform locally.** Archive/max Text blocks run BWT trials to turn
  long-range word repeats into local runs. Level `-3` skips those trials.
- **Match.** Wire uses a hash-chain LZ77. Archive/max/general use DP-LZP
  before entropy coding.
- **Mix or entropy-code.** Slow bit CM blends many context models; Fast/General
  use byte CM + interleaved rANS; Wire optionally wraps LZ tokens in order-0
  rANS.

Stages are modular: swap an engine, re-run `scripts/bench_vs_sota.sh`, keep or
discard based on dual-axis scorecards.

## Projected use cases

- **Wire / `-1`** — fast path for less-redundant or latency-sensitive data.
- **General / `-3`** — gzip-class work without BWT trial cost.
- **Archive / `-9`** — text-centric archival (dickens, webster, source trees,
  JSON) where hybrid ratio vs `zstd -19` matters.
- **Max / `-19`** — Slow bit CM everywhere; binary/`mr` ratio stretch.
- **Learning reference** — readable pipeline of LZ, BWT, LZP, and mixers.

## The method

| Flag | Level | Engine |
|------|------:|--------|
| `--level 1` / `--mode wire` | `-1` | Hash-chain LZ77 (64 KiB window, lazy match) + optional order-0 rANS. Copy if the block does not shrink. |
| `--level 3` / `--mode general` | `-3` | Byte CM + DP-LZP; no BWT/XWRT trials; Exec still uses E8E9. |
| `--level 9` / `--mode hybrid` (default) | `-9` | Fast byte CM on Text/Random; Slow bit CM + DP-LZP on Binary/Exec. |
| `--level 19` / `--mode slow` | `-19` | Bit-level CM with a two-level 16k-bank mixer on every block. |
| `--mode fast` | (not numbered) | Byte CM **with** BWT trials on every block. |

Input is split into variable-size blocks by data type. Each block is classified
by a cheap order-0 Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim — no prediction cost (except level `-1`,
  which still tries wire LZ and copies if it expands).
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range
  word repeats into local runs).
- `Binary` / `Exec` blocks use **1 MiB** chunks (with up to 4 MiB of prior
  same-kind bytes as DP-LZP match history across block boundaries).
- Level `-1` uses 4 MiB independent chunks (no global XWRT dictionary).

The bit-level path runs an online logistic mixer hierarchy:

1. **Bank mixers** (16384 instances): selected by a context hash of byte-class,
   bit-position, order-1/order-2 bytes, and word-hash. Each bank specializes
   weights to its context, avoiding the ~50% saturation a single mixer hits
   on repetitive corpora.
2. **Global mixer**: a context-agnostic fallback over the same models.
3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.

Only the selected bank + global + master are trained per bit — never all 16384.
At block boundaries, weights are decayed (not reset), preserving learned
structure across the stream. The fused probability drives an rANS bit coder
(via the audited [`ans`](https://crates.io/crates/ans) crate).

Because modeling is causal, the decoder reconstructs identical model state from
the coded stream, so round-trips are lossless. Decompress also checks per-block
CRC32 and rejects truncated containers, overlong block payloads, and corrupt
JSON/CSV/XML/XWRT structured transforms (hard error — not an empty output).

## Build

```bash
cargo build --release --bin rcn
```

## Usage

```bash
# Help and version
rcn -h
rcn --version

# Compress a file (default level -9 / hybrid)
rcn compress input.bin output.rcn

# Numbered levels (1/3/9/19 or -1/-3/-9/-19)
rcn compress --level 1 input.bin output.rcn
rcn compress --level 3 input.bin output.rcn
rcn compress --level 19 input.bin output.rcn

# Per-block method/size on stderr
rcn compress --verbose input.bin output.rcn

# Engine aliases
rcn compress --mode wire input.bin output.rcn
rcn compress --mode general input.bin output.rcn
rcn compress --mode fast input.bin output.rcn
rcn compress --mode slow input.bin output.rcn

# Compress a folder into one archive (recursive regular files)
rcn compress mydir out.rcn

# Decompress: omit OUTPUT to restore the stored name (or folder tree)
rcn decompress output.rcn
rcn decompress out.rcn
rcn decompress out.rcn /tmp/restore
rcn decompress output.rcn restored.bin --force

# Benchmark one engine
rcn bench path/to/corpus
rcn bench --level 1 path/to/corpus

# Run the full test suite and report PASS/FAIL
rcn self-test
```

A section-1 man page lives at [`man/rcn.1`](man/rcn.1). Preview it from the
source tree with `man ./man/rcn.1`. To install it system-wide (optional):

```bash
install -m 644 man/rcn.1 "$(manpath | cut -d: -f1)/man1/rcn.1"
```

`cargo install` installs the binary only, not the man page.

`compress` and `decompress` accept a **file or a folder**. A directory is
packed into one `.rcn` archive (recursive regular files; relative paths under
the folder’s basename). Empty directories, permissions, and timestamps are not
stored. `decompress` writes the stored name next to the `.rcn` for a single
file, or restores the tree under the current directory (or `OUTPUT`). v1
containers have no stored name: pass `OUTPUT`, or name the input `*.rcn` so
the suffix can be stripped. Existing files are not overwritten unless
`--force`.

### Installation

`rcn` is installed via `cargo install --locked rcn` or downloaded as a binary
release from [crates.io](https://crates.io/crates/rcn). The package requires
Rust toolchain ≥1.85 (`rust-version` in `Cargo.toml`) and an x86_64 (AVX2) or
arm64 (NEON) processor. On x86_64 (Linux), the AVX2 code path can be disabled
at build time with `--no-default-features --features no_avx2` — see
[CI / testing](DEVELOPMENT.md#ci--testing).

### Streaming / `--stdout`

The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Compress accepts `--verbose` for per-block method and size.

## Stability (RCN1)

Magic `RCN1`. Layout is documented in [`src/container.rs`](src/container.rs).

v1 (library `compress*`):  
`[MAGIC 4][Header 7][optional global dict][BlockEntry × N][payloads…]`

v2 (CLI `compress`, `compress_archive`): same header and block-entry layout,
plus a member table after the optional global dict so decompress can restore
the original file or folder paths.

RCN1 v1 remains readable. Unknown methods fail closed. New engines add method
bytes; they do not change the header or block-entry layout. Old decoders
reject v2 (`unsupported version`).

## License

MIT — see [LICENSE](LICENSE).
