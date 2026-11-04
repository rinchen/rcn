# Rcn measured benches

Raw ratio and throughput numbers. Win/lose charts live in
[README.md](README.md#scorecards).

Host: release `rcn`, macOS, Apple Silicon (ARM64, NEON `walk_dist`),
2026-11-03, `.work/bench5/` refreshed with updated wire LZ chain (T1, CHAIN 4→8),
Binary DP-LZP match window (T2, 4→8 MiB), General BWT trials (T3),
PGO (T4, unblocked on Apple Silicon via llvm-profdata from rustup),
Binary Indirect (W5, added to binary/exec stacks), Order-12 PPMd (C10, tested, no improvement).
Level `-9`/`fast`/`-19` ratios and speeds match the 2026-09-11 W1–W7 refresh
unless noted. Level `-1`/`-3` and peer cmp MB/s from 2026-09-12 (README).
Ratios from the 2026-09-11 `SKIP_RCN=1` pass.

```bash
rcn bench .work/bench5
rcn bench --level 1 .work/bench5
rcn bench --level 3 .work/bench5
scripts/bench_vs_sota.sh .work/bench5
SKIP_RCN=1 scripts/bench_vs_sota.sh .work/bench5
```

`ratio%` is compressed size as a percentage of the original (lower is
better). Speed is MB/s of original bytes (higher is better).

A/B history: [DEVELOPMENT.md](DEVELOPMENT.md).

## Level `-1` (`--mode wire`)

Hash-chain LZ + optional order-0 rANS.

| file | orig (KB) | rcn-1 ratio% | cmp MB/s | dec MB/s |
|------|----------:|-------------:|---------:|---------:|
| dickens | 9953.6 | 45.2 | 68.5 | 50.1 |
| webster | 40487.0 | 35.1 | 181.40 | 74.51 |
| nci | 32767.0 | 11.9 | 123.24 | 203.00 |
| mr | 9736.9 | 39.6 | 82.44 | 60.49 |
| json | 478.5 | 0.1 | 277.55 | 1185.48 |

## Level `-3` (`--mode general`)

Byte CM + DP-LZP, no BWT trials.

| file | orig (KB) | rcn-3 ratio% | cmp MB/s | dec MB/s |
|------|----------:|-------------:|---------:|---------:|
| dickens | 9953.6 | 26.9 | 12.18 | 3.92 |
| webster | 40487.0 | 33.9 | 37.09 | 5.55 |
| nci | 32767.0 | 14.2 | 53.68 | 17.44 |
| mr | 9736.9 | 31.4 | 4.36 | 5.22 |
| json | 478.5 | 0.5 | 2.77 | 168.37 |

## Level `-9` (`--mode hybrid`, default)

Text/Random → Fast byte CM (global XWRT + DP-LZP literal-skip); Binary/Exec →
Slow bit CM + DP-LZP.

| file | orig (KB) | rcn-9 ratio% | cmp MB/s | dec MB/s | zstd -19 ratio% |
|------|----------:|-------------:|---------:|---------:|----------------:|
| dickens | 9953.6 | 26.5 | 3.23 | 5.00 | 28.0 |
| webster | 40487.0 | 20.1 | 6.62 | 7.00 | 20.9 |
| nci | 32767.0 | 5.0 | 8.29 | 12.25 | 5.0 |
| mr | 9736.9 | 27.3 | 0.03 | 0.03 | 31.2 |
| json | 478.5 | 0.1 | 14.33 | 113.70 | 0.0 |

## Fast path (`--mode fast`, not a numbered level)

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -19 ratio% |
|------|----------:|-----------:|---------:|---------:|---------------:|
| dickens | 9953.6 | 26.5 | 3.26 | 5.10 | 28.0 |
| webster | 40487.0 | 20.1 | 6.67 | 6.75 | 20.9 |
| nci | 32767.0 | 5.0 | 8.29 | 11.95 | 5.0 |
| mr | 9736.9 | 31.5 | 4.31 | 5.23 | 31.2 |
| json | 478.5 | 0.1 | 14.14 | 116.74 | 0.0 |

## Level `-19` (`--mode slow`)

Text/`nci` slow ratios: dickens 40.2%, webster 29.3%, nci 7.3%, json 0.1%.
Hybrid/Slow `mr` is 27.3%.

## Peer CLI ratio%

| file | rcn-1 | rcn-3 | rcn-9 | rcn-19 | rcn-fast | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|------:|------:|------:|------:|---------:|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 45.4 | 39.5 | 26.5 | 40.2 | 26.5 | 28.0 | 27.8 | 27.7 | 27.7 | 37.8 | 41.8 |
| webster | 35.1 | 33.9 | 20.1 | 29.3 | 20.1 | 20.9 | 20.2 | 20.3 | 20.3 | 29.1 | 33.0 |
| nci | 11.9 | 14.2 | 5.0 | 7.3 | 5.0 | 5.0 | 5.2 | 4.5 | 4.5 | 8.9 | 11.0 |
| mr | 39.6 | 31.4 | 27.3 | 27.3 | 31.5 | 31.2 | 27.6 | 28.3 | 28.3 | 36.7 | 38.3 |
| json | 0.1 | 0.5 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 |

## Peer CLI compress MB/s

From the 2026-09-11 `SKIP_RCN=1` refresh.

| file | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 2.5 | 2.2 | 0.8 | 21.1 | 53.1 | 208.0 |
| webster | 3.2 | 2.1 | 0.8 | 31.0 | 241.8 | 693.8 |
| nci | 3.5 | 6.1 | 0.9 | 22.9 | 278.4 | 876.3 |
| mr | 4.3 | 3.7 | 0.7 | 11.5 | 46.6 | 304.2 |
| json | 19.6 | 18.3 | 16.1 | 20.8 | 21.3 | 20.6 |