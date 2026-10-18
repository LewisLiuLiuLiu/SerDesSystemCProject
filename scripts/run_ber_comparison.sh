#!/bin/bash
# ============================================================================
# 2026-10: BER comparison for the peters channel (nrz_ber_tb + offline PRBS31)
#
# Cases:
#   simple    - built-in 6 dB / 20 GHz channel model
#   old_diff  - original state-space model, differential port mapping
#   new_diff  - delay-aware per-pair VF refit (e=0), differential mapping
#
# The differential mapping for this 4-port fixture (interleaved by polarity):
#   TX pair = physical ports 0/2, RX pair = physical ports 1/3
#
# For the paper-style A2 operating point (TX FFE + fixed phase + noise):
#   scripts/run_ber_comparison.sh 100000 config/ts4_a2_ber_diff.json
#
# Usage: scripts/run_ber_comparison.sh [UI_COUNT] [extra --config overlay]
# ============================================================================
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
UI="${1:-20000}"
OVERLAY="${2:-}"
OUT="${OUT:-/tmp/ber_cmp}"
mkdir -p "$OUT"

run_case() {
    local name="$1"; shift
    echo "=== $name ==="
    # shellcheck disable=SC2086
    "$BUILD/bin/nrz_ber_tb" "$@" ${OVERLAY:+--config "$OVERLAY"} -d "$UI" \
        -o "$OUT/$name" > "$OUT/$name.log" 2>&1
    python3 "$ROOT/scripts/ts3/ber_from_decisions.py" \
        --decisions "$OUT/${name}_decisions.csv" --out "$OUT/${name}_ber.json" >/dev/null
    python3 - "$OUT/${name}_ber.json" "$name" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  lag={r['lag']} inv={r['invert']} BER={r['ber']:.3e} "
      f"(errors {r['errors']}/{r['n_compared']})")
PY
}

run_case simple
run_case old_diff ss "$ROOT/config/peters_ss_channel_diff.json"
run_case new_diff ss "$ROOT/config/peters_ss_vf12_diff.json"

echo
echo "Results in $OUT (decisions CSV, BER JSON, TB logs)"
