#!/bin/zsh
# Head-to-head on one track: wall time and peak RSS of complete runs
# (decode + separate + write), each tool alone on the machine, one after
# another. Cold rows are whole CLI processes measured with /usr/bin/time -l
# (macOS, "maximum resident set size" in bytes). Resident rows keep the
# model loaded: `charon serve` jobs (client wall time; the server does all
# the work) and PyTorch through demucs.api in one process.
#
# Usage: compare_tools.sh <track.wav> <out dir> <charon bin> <htdemucs_split.onnx> \
#            <htdemucs_split_coreml.onnx> <stem-splitter bin> <demucs-rs bin> <python with demucs>
# First runs of tools that download models on first use are discarded
# (run each tool once before).
set -e
track=$1; out=$2; charon=$3; model_cpu=$4; model_coreml=$5; ssc=$6; drs=$7; py=$8
mkdir -p "$out"
run() {
  local name=$1; shift
  echo "== $name"
  /usr/bin/time -l "$@" > "$out/$name.log" 2> "$out/$name.time" || echo "FAILED: $name"
  awk '/real/ {printf "  wall %s s", $1} /maximum resident/ {printf "  peak RSS %.0f MB\n", $1/1048576}' "$out/$name.time"
}
run charon_cpu_cold     "$charon" separate "$track" -o "$out/charon_cpu" --model "$model_cpu" --ep cpu --no-server
run charon_coreml_cold  "$charon" separate "$track" -o "$out/charon_coreml" --model "$model_coreml" --ep coreml --no-server
run stem_splitter       "$ssc" split -i "$track" -o "$out/stem_splitter" -q
run demucs_rs           "$drs" "$track" -o "$out/demucs_rs"
run torch_cpu           "$py" -m demucs -n htdemucs --shifts 0 -d cpu -o "$out/torch_cpu" "$track"
run torch_mps           "$py" -m demucs -n htdemucs --shifts 0 -d mps -o "$out/torch_mps" "$track"

sock=/tmp/charon-compare.sock
echo "== charon_coreml_resident (server load, then 3 jobs)"
"$charon" serve --model "$model_coreml" --ep coreml --socket "$sock" > "$out/serve.log" 2>&1 &
for i in {1..90}; do "$charon" ping --socket "$sock" > /dev/null 2>&1 && break; sleep 1; done
grep "model loaded" "$out/serve.log"
for i in 1 2 3; do
  /usr/bin/time -l "$charon" separate "$track" -o "$out/charon_resident" --socket "$sock" 2>&1 | awk '/stems ->/ {sub(/.*\(/, "("); print "  " $0} /real/ {printf "  client wall %s s\n", $1}'
done
ps -o rss= -p "$(pgrep -f 'charon serve' | head -1)" | awk '{printf "  server RSS %.0f MB\n", $1/1024}'
"$charon" stop --socket "$sock" > /dev/null
echo "== torch_mps_resident"; "$py" "$(dirname "$0")/torch_resident.py" "$track" "$out/torch_resident_mps" mps 3 2>&1 | grep -v Warn | sed 's/^/  /'
echo "== torch_cpu_resident"; "$py" "$(dirname "$0")/torch_resident.py" "$track" "$out/torch_resident_cpu" cpu 2 2>&1 | grep -v Warn | sed 's/^/  /'
