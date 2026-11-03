# Rcn development notes

Optimization tickets, A/B history, CI/testing notes, and roadmaps.
For product overview and win/lose charts, see [README.md](README.md).
Measured numbers: [BENCHMARKS.md](BENCHMARKS.md).
Ticket tables also live in [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

Last updated: 2026-09-12. Test status: see `cargo test --lib` / CI. Headline
benches: W1–W7 on 2026-09-11; levels `-1`/`-3` + dual scorecards on
2026-09-12 (see [BENCHMARKS.md](BENCHMARKS.md)).

---

## Completed optimization checklist

Closed through C11 / S12 (C10 tried and reverted), plus V1–V3 / R1–R2
(2026-09-11). See [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

### Compression - beat zstd -19

- ✅ **Promote Order-8 PPMd with SEE + sparse de Bruijn to default** – WordModel + banks + 32MB window + XWRT
- ✅ **Compress the match side-stream (varint)** – delta_pos/len/dist varints
- ✅ **Global XWRT dictionary (top-128)** – corpus-wide; C5 measured dickens −4.6pt, webster −4.0pt
- ✅ **C6 — Global XWRT 128 → 512 with ESC tokens** – `0x80..=0xFE` for ids 0..126; `0xFF || u16_le(id)` for 127..511
- ✅ **C7 — Adaptive DP LZP threshold** – Text ≥12; Binary/Exec/Random stay 16
- ✅ **C8 — 16k banks** – `NUM_MIXERS=16384`; Indirect left out of default Text stack
- ✅ **C9 — FSE-family match side-stream** – order-0 byte rANS on the varint blob when smaller
- ❌ **C10 — Order-12 Text PPMd** – reverted (−0.003pt)
- ✅ **C11 — CSV/XML stream splitting** – detectors + BWT trial wiring
- ✅ **R1 — Hybrid mode** – Fast Text/Random, Slow Binary/Exec; mixed methods in one container
- ✅ **R2 — Fast nci** – global XWRT on Fast trials + DP-LZP on byte path → nci **4.97%**
- ✅ **Default → hybrid** after full headline gate cleared
- ✅ **L1 / L3 / L9 / L19** — numbered levels + wire/general engines (2026-09-12)
- ✅ **D1** — dual ratio + cmp-speed + both scorecards

### Speed

- ✅ **S1–S4, S7, S8–S12** – SIMD walk_dist, SoA, stretch, mimalloc, interleaved rANS, libsais optional, classify-ahead, prefetch, parallel trials
- ✅ **V1 — BWT trial payload cache** – no double-encode of winner
- ✅ **V2 — aarch64 NEON `walk_dist`** – parity tests vs scalar
- ✅ **V3 — `release-prof` + enum `StackModel`** – symbol-friendly profile; Slow hot path without `dyn BitModel`
- ✅ **W1 — Binary 1 MiB + cross-block match hist** – hybrid mr 27.3%
- ✅ **W2 — Fast Binary order-3** – fast mr 35.1%→31.5%
- ✅ **W3 — Parallel Fast Text/Random encode** – large Text cmp MB/s uplift
- ✅ **W4 — XWRT-1024** – nci ~4.99%
- ❌ **W5 — Binary-only Indirect** – killed (mr regression)
- ✅ **W6 — Kind-specific DP costs** – Binary/Exec avg_bits 5.0
- ✅ **W7 — libsais default + PGO recipe + cold error helpers**

Apple Silicon (arm64) uses the NEON `walk_dist` path by default. CI still runs
`no_avx2` for the x86_64 scalar gate. `bwt_libsais` is on by default.

---

## CI / testing

GitHub Actions runs on `ubuntu-latest` (x86_64). The AVX2 SIMD path in
`bytecodec` is compiled there, but CI runs tests with the `no_avx2` feature
(scalar path only) for a stable Linux gate. Local builds still use AVX2 by
default on x86_64. Apple Silicon (arm64) uses NEON `walk_dist` by default.
AVX2↔scalar and NEON↔scalar bit-identity are covered by unit tests.

```bash
cargo test --lib
cargo test --lib --features no_avx2       # CI-like scalar path (libsais still on by default)
cargo build --profile release-prof        # symbols for sample/Instruments
rcn self-test                             # wraps cargo test --lib
```

Pre-commit: `.pre-commit-config.yaml` runs `cargo build` +
`cargo test --features no_avx2` (mirrors CI).

### Profile-guided optimization (W7)

Release builds already use fat LTO + `codegen-units=1`. For an extra ~5–15% on
hot loops, build with PGO (keep CI non-PGO):

```bash
# Requires cargo-pgo: cargo install cargo-pgo
cargo pgo build -- --release --bin rcn
# Instrument / run a short corpus, then:
./target/release/rcn compress --mode hybrid path/to/dickens /tmp/out.rcn
./target/release/rcn decompress /tmp/out.rcn /tmp/out.bin
cargo pgo optimize -- --release --bin rcn
```

`bwt_libsais` is now a **default** feature (W7); disable with
`--no-default-features --features two_pass` if needed.

## Dual-axis backlog (2026-09-12)

North star is no longer ratio-only. Each level must beat its peer class on
**ratio and compress MB/s**. See [README.md](README.md#goal).

| Level | Status on headline set |
|------:|------------------------|
| `-9` vs `zstd -19` | both-win text/`nci`; **split** `mr` (ratio win, 0.03 MB/s) and json |
| `-1` vs `lz4 -9` / `zstd -1` | both-win json/`mr` vs lz4; **lose cmp** vs `zstd -1` almost everywhere |
| `-3` vs `gzip -9` | no both-win yet (ratio or speed splits) |
| `-19` | Slow text still loses `-19` on ratio; `mr` leads peers on ratio |

Do not reopen Text Indirect / Binary DP≥24 / Order-12 without ≥0.3pt evidence.
Binary-only Indirect was A/B'd (W5) and **killed** (mr regression).

Next levers: **faster `-1` wire LZ (chain depth 3→4 in wire.rs)**; PGO build for `-9` binary; `-3` BWT budget vs gzip.

---

## DP-optimal LZP parse (historical measure)

DP optimal LZP parse runs a forward LZP match pre-pass and emits `(len, dist)`
records for matches ≥ 16 bytes (Text now gates at ≥12). Matched bytes are
skipped in the rANS stream — only literals are CM-encoded. The match side-stream
uses varint-encoded records (delta_pos + len + dist), optionally entropy-coded (C9).

| file | orig (KB) | rcn ratio% | vs default | zstd -1 ratio% | beats zstd-1? |
|------|----------:|-----------:|-----------:|---------------:|:------------:|
| dickens | 9953.6 | 41.9 | 46.2→41.9 (**−4.3pt**) | 41.7 | ~parity |
| webster (10MB) | 10000.0 | 31.4 | 35.1→31.4 (**−3.7pt**) | 33.0 | ✅ |
| nci | 32767.0 | 8.2 | 9.0→8.2 (**−0.8pt**) | 85.2 | ✅ |
| mr | 9736.9 | 29.1 | 27.5→29.1 (**+1.6pt**) | 38.3 | ✅ (vs default) |
| json | 478.5 | 0.1 | 0.1 (same) | 0.3 | ✅ |
| huge_json | 5641.7 | 1.3 | 2.7→1.3 (**−1.4pt**) | 2.5 | ✅ |
| massive_json | 22885.9 | 0.97 | 0.81→0.97 (**+0.16pt**) | 2.48 | ✅ |

**Key insight:** DP optimal parse with literal-skipping rANS is a net win on most
files. `mr` and `massive_json` regressed slightly when SSM was co-introduced; DP
is default and SSM is isolated from DP.

---

## A/B history (2026-09)

### Architecture & modeling

| change | files tested | result | action |
|---|---|---|---|
| Per-bit-position mixer context | mr, dickens, json, webster, nci | **improved all 5** | kept as default |
| Classifier-aware method bytes | mr, dickens, json, webster, nci | neutral | kept as infrastructure |
| Word/string model (case-folded, bigram prefix) | dickens, json, webster, nci | +0.1pt on 4/5 | kept as default (text blocks) |
| Refined word model (trigram + char-class + 21-bit table) | json | regressed 5.6%→5.8% | reverted to simple word model |
| Record segmentation model (JSON key/value parser) | dickens, json, webster, nci | neutral json/dickens/mr; regressed webster/nci | reverted |
| ICM (22-state PAQ8) | mr, dickens, json, webster, nci | regressed on 4/5 | reverted |
| ICM (256-state probability-quantized) | mr, dickens, json, webster, nci | regressed on all 5 | reverted |
| Order-4 PPM with word-boundary-aware context masking | mr, dickens, json, webster, nci | regressed dickens +0.1pt, webster +0.5pt, nci +0.1pt, json +3.6pt | reverted |
| Lazy multi-context LZP (hash chains + longest-match) | mr, dickens, json, webster, nci | neutral (−0.1pt) | kept in place, not adopted |
| Two-pass CM residual (match records + CM literals) | mr, dickens, json, webster, nci | nci +2.7pt, json/webster regressed | reverted; match overhead too high at 64 KiB |
| Literal bypass hint model (high-entropy byte bypass) | mr, dickens, json, webster, nci | regressed dickens 56.3%→57.1% | reverted |
| **Context-selected 8k mixer banks** (8192 per-context LogisticMixer instances selected by byte-class + order-1/order-2 + word-hash, blended with global + master) | mr, dickens, json, webster, nci | **improved**: json 3.9%→3.0%, dickens 51.7%→51.2% | **kept as default** — later expanded to 16k banks (C8) |
| **Indirect context + DMC models** | mr, dickens, json, webster, nci | regressed dickens +0.7pt, webster +0.4pt; json improved | reverted; IndirectModel remains in-tree, not default |
| **Cross-block persistence + real 4MB LDM window** | json, mr, dickens, nci, webster | **improved**: json 5.5%→3.9%, mr 28.6%→27.3%, dickens 56.0%→51.7%, nci 26.6%→20.9%, webster 50.4%→45.1% | **kept as default** |
| LZP ring buffer performance fix (O(n) drain→O(1) ring) | all files | performance fix, no ratio change | kept |
| Micro SSM mixer (16-dim recurrent state replacing logistic mixer) | json, mr | **regressed**: json 3.9%→10.7%, mr 27.3%→37.2% | reverted; SSM too large for 64KB blocks, gradient issues |
| Second-order mixer training (Adam + per-model lr_scale) | dickens, mr, json | **neutral** (Adam) / **neutral** (SGD + lr_scale) | kept as default; Adam never measurably better on default stacks |
| BWT text trial (RawCm vs BWT→MTF→RLE0→CM vs LZP→BWT→MTF→CM) | dickens, json, webster | **improved**: json 3.0%→0.1%, dickens 51.2%→46.2%, **webster 50.4%→35.1%** | **kept as default** |
| JSON stream splitting + per-stream pipeline selection | json (478KB–22MB) | **improved**; beats zstd-1 and zstd-19 on large JSON | **kept as default** |
| Order-8 PPMd with SEE + sparse de Bruijn | webster, dickens, json | mixed; config now matches hybrid_ppm3 | **promoted to default** |
| DP optimal LZP parse (now default) | dickens, webster, nci, mr, json, huge_json, massive_json | **improved** on 5/7 (dickens −4.3pt, webster −3.7pt); **regressed** mr +1.6pt, massive_json +0.16pt | **kept as default** — SSM isolated to avoid regression |
| **Exec E8E9 transform** | Exec executables | Converts x86 relative offsets to absolute (3-5pt on Exec corpora) | **kept as default** |
| **XWRT dictionary before BWT** | Text blocks | Build top 2k words per block, replace with tokens, then BWT→MTF→RLE0→CM | **per-block superseded by global** |
| **Global XWRT dictionary** | dickens, webster, nci, mr, json | Top-128 then C6 top-512 with ESC; measured C5 dickens −4.6pt, webster −4.0pt; C6 ≈−1.1pt dickens vs HEAD | **kept as default** |
| **SSE/APM/APM2 cascade** | mr, dickens, json, webster, nci | **improved all 5** | **wired into codec** |
| **AVX2 SIMD walk_dist** | All | 5-10x fast path speedup, zero ratio loss | **completed** |
| **C10 Order-12 PpmdSsm (Text)** | dickens 2MB | −0.003pt (990101→990035 B) | **reverted**; API `with_max_order` kept |
| **C11 CSV/XML stream split** | `testdata/structured/sample.csv` (701KB), `sample.xml` (586KB) | CSV: trial picks `CsvSplit` (payload 156KB vs RawCm 702KB / BWT 201KB); slow **1.02%**, fast **0.95%** (methods 15/17). XML: payload trial prefers `XmlSplit`; slow often lands global XWRT (method 13, **1.52%**); fast XML-split **2.09%**. Need ≥256KB for trials. | **kept** |
| **S11 MixerAcc stretch + bank prefetch** | dickens 200KB | deterministic compress; ratio-neutral by design | **kept** |
| **S12 parallel_map_sizes BWT trials** | (infra) | RawCm/XWRT/JSON/CSV/XML size jobs via nested `rayon::join` | **kept**; ratio unchanged |
| **V1 BWT trial payload cache** | dickens/text | winner payload returned; no re-encode | **kept** |
| **R1 Hybrid mode** | headline 5 | clears zstd -19 on all five | **default** |
| **R2 Fast XWRT + DP-LZP** | nci | 5.1%→4.97% | **kept** |
| **V2 NEON walk_dist** | aarch64 | bit-identical to scalar | **kept** |
| **V3 StackModel enum** | slow path | removes `dyn BitModel` in hot loop | **kept** |
| **T1 Wire LZ chain depth 3→4** | wire test data | neutral small / small ratio lift on repetitive | kept |
| **T2 Binary DP-LZP window 4→8 MiB** | mr, nci, binary blobs | small ratio lift | kept |
| **T3 General mode (-3) BWT trial ≥256KB** | large text | ratio lift vs plain byte CM | kept |

### Speed passes

| pass | description | files tested | result | status |
|---|---|---|---|---|
| #1 | LazyLzp removal (O(n²) memmove per byte over 1MB; match-extension loop dead). Model rewritten with fixed-capacity ring buffer + causal extension loop. | dickens, webster | **3× encode speedup on text, zero ratio change** (dickens 2MB: 10.9s→3.3s, byte-identical) | removed from all default stacks |
| #2 | Byte-level "fast" path (`--mode fast`): PPM-style single-context count coder (deterministic order-0/1/2 selector + fused 256-symbol cumulative walk_dist + byte rANS). No mixer/softmax. | dickens 2MB | **2.6× encode speedup vs slow with ~1.7× better ratio on BWT+MTF streams** | kept as `--mode fast` |
| #3 | Single-pass acc-merge across the mixer chain + Q16 fixed-point mixer | dickens 2MB (slow) | **~10% encode speedup, ratio flat** | kept as default |
| #4 | 32-way interleaved byte rANS | dickens 2MB (fast) | **no measurable speedup** while serial `walk_dist` dominated; later wired + re-benched after AVX2 `walk_dist` | **wired into the fast path** |
| #5 | macOS `sample` (release, stripped) | dickens | Slow: bit CM/mixer call chain dominates; S11 prefetch not a separate frame. Fast: BWT/`walk_dist` — next speed lever there. | note only |

### Code-quality / correctness notes

| change | files tested | result | action |
|---|---|---|---|
| round-trip verification | all 5 | lossless | every pass round-trip verified |
| test suite | all | 198+ green (`cargo test --lib`) | kept |
| bit-identical output | dickens 2MB | each pass `cmp`-identical to prior where claimed | kept |
| container / BWT hardening (2026-09-11) | unit + Fast/CSV/XML round-trips | corrupt payloads/`comp_len`/dict → `RcnError`; AVX2↔scalar parity | kept |
| orphan delta transform | — | never wired into codec | **deleted** (`src/model/delta.rs`) |
