#!/bin/bash
# E1: o5 LLM proxy (Qwen3.5-35B-A3B Q8_0, same shapes as o5 backbone) on H20 via llama-batched-bench
set -u
BB=/data/codex-qwen-complete-20260902/build-cuda/bin/llama-batched-bench
M=/data/codex-qwen-complete-20260902/models/Qwen3.5-35B-A3B-Q8_0.gguf
OUT=/data/wangkaiqi/o5cap/e1; mkdir -p $OUT
run() { # name, CUDA_VISIBLE_DEVICES, args...
  local name=$1; local devs=$2; shift 2
  echo "[$(date +%T)] start $name devs=$devs :: $*" | tee -a $OUT/progress.log
  CUDA_VISIBLE_DEVICES=$devs $BB -m $M -ngl 99 -fa on "$@" > $OUT/$name.log 2>&1
  echo "[$(date +%T)] end $name rc=$?" | tee -a $OUT/progress.log
}
# --- track 1 (GPU2): single-card timing sweeps
(
  run A_single_pp17_100_tg16 2 -c 32768 -np 256 -b 4096 -ub 512 -npp 17,100 -ntg 16 -npl 1,2,4,8,16,32,64,128,256
  run E_single_longctx_pp4000 2 -c 300000 -np 64 -b 4096 -ub 512 -npp 4000 -ntg 8 -npl 1,16,64
  run B_mem_np128_ctx8k_f16 2 -c 1048576 -np 128 -npp 16 -ntg 1 -npl 1
  run B_mem_np128_ctx8k_q8 2 -c 1048576 -np 128 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 1
  run B_mem_np256_ctx4k_f16 2 -c 1048576 -np 256 -npp 16 -ntg 1 -npl 1
  run B_mem_np1_ctx8k_f16 2 -c 8192 -np 1 -npp 16 -ntg 1 -npl 1
) &
# --- track 2 (GPU4-7): layer-split timing
(
  run C_split4_pp100_tg16 4,5,6,7 -sm layer -c 32768 -np 256 -b 4096 -ub 512 -npp 100 -ntg 16 -npl 1,8,32,64,128,256
  run D_split2_pp100_tg16 4,5 -sm layer -c 32768 -np 256 -b 4096 -ub 512 -npp 100 -ntg 16 -npl 1,8,32,64,128,256
) &
wait
echo "[$(date +%T)] ALL DONE" | tee -a $OUT/progress.log
