#!/bin/bash
set -u
BB=/data/codex-qwen-complete-20260902/build-cuda/bin/llama-batched-bench
M=/data/codex-qwen-complete-20260902/models/Qwen3.5-35B-A3B-Q8_0.gguf
OUT=/data/wangkaiqi/o5cap/e2; mkdir -p $OUT
run() { local name=$1; shift
  echo "[$(date +%T)] start $name :: $*" | tee -a $OUT/progress.log
  ( while true; do nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > $OUT/$name.mem &
  local sp=$!
  CUDA_VISIBLE_DEVICES=2 $BB -m $M -ngl 99 -fa on --verbose "$@" > $OUT/$name.log 2>&1
  kill $sp 2>/dev/null
  echo "[$(date +%T)] end $name rc=$? peakMiB=$(sort -n $OUT/$name.mem | tail -1)" | tee -a $OUT/progress.log
}
run np1_ctx8k_f16     -c 8192    -np 1   -npp 16 -ntg 1 -npl 1
run np128_ctx8k_f16   -c 1048576 -np 128 -npp 16 -ntg 1 -npl 1
run np128_ctx8k_q8    -c 1048576 -np 128 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 1
run np256_ctx8k_f16   -c 2097152 -np 256 -npp 16 -ntg 1 -npl 1
run np256_ctx8k_q8    -c 2097152 -np 256 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 1
run np256_ctx32k_q8   -c 8388608 -np 256 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 1
run np256_ctx8k_q8_ub128 -c 2097152 -np 256 -ctk q8_0 -ctv q8_0 -ub 128 -b 2048 -npp 16 -ntg 1 -npl 1
echo "[$(date +%T)] ALL DONE" | tee -a $OUT/progress.log
