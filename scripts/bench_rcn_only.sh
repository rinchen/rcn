#!/usr/bin/env bash
#
# bench_rcn_only.sh — rcn-only benchmark on a corpus directory.
# Outputs one row per file per level + a W-L-T tally for each axis.

set -u

CORPUS="${1:-}"
RCN="${2:-./target/aarch64-apple-darwin/release/rcn}"

if [[ -z "$CORPUS" || ! -d "$CORPUS" ]]; then
    echo "usage: $0 <corpus_dir> [rcn_bin]" >&2
    exit 1
fi
if [[ ! -x "$RCN" ]]; then
    echo "building rcn (release)..." >&2
    cargo build --release --bin rcn || { echo "rcn build failed" >&2; exit 1; }
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

now() { python3 -c 'import time; print("%.6f", time.time())'; }
run_and_time() {
    local t0 t1
    t0="$(now)"
    "$@" >"$out" 2>/dev/null
    t1="$(now)"
    awk -v b="$t0" -v b2="$t1" 'BEGIN { printf "%.6f", b - b2 }'
}
mbps() { awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.1f", (b/1e6)/s; else printf "0.0" }'; }
verdict_ratio() { awk -v a="$1" -v b="$2" 'BEGIN { if (a == b || (a < 0.5 && b < 0.5)) print "tie"; else if (a < b) print "win"; else print "lose" }'; }
verdict_speed() { awk -v a="$1" -v b="$2" 'BEGIN { if (a == b) print "tie"; else if (a > b) print "win"; else print "lose" }'; }

sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'; }
set_metric() { eval "${kind}_$(sanitize "$2"):-}"; }
get_metric() { eval "printf '%s' \"\${${kind}_$(sanitize "$2"):-}\""; }

# Corpus files
CORPUS_FILES=()
for f in "$CORPUS"/*; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f")"
    orig="$(wc -c <"$f" | tr -d '[:space:]')"
    [[ "$orig" -eq 0 ]] && continue
    CORPUS_FILES+=("$name")
    [[ "$orig" -eq 0 ]] && continue
    okb="$(awk -v b="$orig" 'BEGIN { printf "%.1f", b/1024 }')"
    clear_metrics

    echo "# $name"
    for mode in rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast; do
        t="$(run_and_time "$WORK/n${mode}.rcn" "$RCN" compress --level $(( ${mode#rcn-} )) "$f" "$WORK/n${mode}.rcn")"
        c="$(wc -c <"$WORK/n${mode}.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n${mode}.out" "$RCN" decompress "$WORK/n${mode}.rcn" "$WORK/n${mode}.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "$mode" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    done

    print_axis_summary ratio "$name"
    print_axis_summary cmp "$name"
    print_axis_summary both "$name"

print_axis_summary ratio "$name" do
    for mode in rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast; do
        rcn_r="$(get_metric RATIO "$mode")"; rcn_c="$(get_metric CMP "$mode")"; [[ -n "$rcn_r" && -n "$rcn_c" ]] || continue
        vr="$(verdict_ratio "$rcn_r" "$rcn_c")"; vc="$(verdict_speed "$rcn_c" "$rcn_r")"
        if [[ "$vr" != "lose" && "$vc" == "win" ]]; then both="win"; elif [[ "$vr" == "lose" && "$vc" == "lose" ]]; then both="lose"; else both="split"; fi
        printf " %s=%s/%s/%s" "$mode" "$vr" "$vc" "$both"
        bump_tally ratio "$mode" "$mode" "$vr"; bump_tally cmp "$mode" "$mode" "$vc"; bump_tally both "$mode" "$mode" "$both"
    done

print_axis_summary cmp "$name" do
    for mode in rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast; do
        rcn_r="$(get_metric RATIO "$mode")"; rcn_c="$(get_metric CMP "$mode")"; [[ -n "$rcn_r" && -n "$rcn_c" ]] || continue
        vr="$(verdict_ratio "$rcn_r" "$rcn_c")"; vc="$(verdict_speed "$rcn_c" "$rcn_r")"
        if [[ "$vr" != "lose" && "$vc" == "win" ]]; then both="win"; elif [[ "$vr" == "lose" && "$vc" == "lose" ]]; then both="lose"; else both="split"; fi
        printf " %s=%s/%s/%s" "$mode" "$vr" "$vc" "$both"
        bump_tally ratio "$mode" "$mode" "$vr"; bump_tally cmp "$mode" "$mode" "$vc"; bump_tally both "$mode" "$mode" "$both"
    done

print_axis_summary both "$name" do
    for mode in rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast; do
        both_r="$(get_metric RATIO "$mode")"; both_c="$(get_metric CMP "$mode")"; [[ -n "$both_r" && -n "$both_c" ]] || continue
        if [[ "$both_r" == "win" ]]; then both="win"; fi
        printf " %s=%s/%s" "$mode" "$both_r" "$both_c"
        bump_tally both "$mode" "$mode" "$both"
    done
}

bump_tally() { local base key v; base=$(sanitize "${axis}_${mode}_${peer}"); case "$v" in win) key="TALLY_${base}_w" ;; lose) key="TALLY_${base}_l" ;; tie) key="TALLY_${base}_t" ;; *) return ;; esac; cur=$(eval "printf '%s' \"\${${key}:-0}\""); eval "${key}=$((cur + 1))"; }

print_axis_summary() {
    local axis="$1" label="$2" mode peer base w l t any=0
    for mode in rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast; do
        for peer in zstd-19 xz-9 brotli-11 gzip-9 lz4-9 zstd-1; do
            base=$(sanitize "${axis}_${mode}_${peer}")
            w=$(eval "printf '%s' \"\${TALLY_${base}_w:-0}\"}")
            l=$(eval "printf '%s' \"\${TALLY_${base}_l:-0}\")"
            t=$(eval "printf '%s' \"\${TALLY_${base}_t:-0}\")"
            (( w + l + t )) || continue
            if (( any == 0 )); then echo; echo "corpus W-L-T ($label):"; any=1; fi
            printf "  %-8s vs %-10s  %d-%d-%d\n" "$mode" "$peer" "$w" "$l" "$t"
        done
    done
}
