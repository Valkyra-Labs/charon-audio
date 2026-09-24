#!/bin/zsh
# Head-to-head on one track: wall time and peak RSS of complete CLI runs
# (decode + separate + write), each tool alone on the machine, one after
# another. Each tool writes its stems so agreement with PyTorch can be
# computed afterwards with compare_outputs.py.
#
# Usage: compare_tools.sh <track.wav> <out dir> <htdemucs.onnx> \
#            <charon separate bin> <stem-splitter bin> <demucs-rs bin> <python with demucs>
#
# Every row is one process measured with /usr/bin/time -l (macOS), which
# reports "maximum resident set size" in bytes. First runs of tools that
# download models on first use are discarded (run the tool once before).
set -e
track=$1; out=$2; model=$3; charon=$4; ssc=$5; drs=$6; py=$7
mkdir -p "$out"
run() {
  local name=$1; shift
  echo "== $name"
  /usr/bin/time -l "$@" > "$out/$name.log" 2> "$out/$name.time" || echo "FAILED: $name"
  awk '/real/ {printf "  wall %s s", $1} /maximum resident/ {printf "  peak RSS %.0f MB\n", $1/1048576}' "$out/$name.time"
}
run charon_lowmem  "$charon" "$track" "$out/charon_lowmem" "$model"
run charon_maxspeed "$charon" "$track" "$out/charon_maxspeed" "$model" --max-speed
run stem_splitter  "$ssc" split -i "$track" -o "$out/stem_splitter" -q
run demucs_rs      "$drs" "$track" -o "$out/demucs_rs"
run torch_cpu      "$py" -m demucs -n htdemucs --shifts 0 -d cpu -o "$out/torch_cpu" "$track"
run torch_mps      "$py" -m demucs -n htdemucs --shifts 0 -d mps -o "$out/torch_mps" "$track"
