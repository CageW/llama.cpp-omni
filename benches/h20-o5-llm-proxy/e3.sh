#!/bin/bash
set -u
BB=/data/codex-qwen-complete-20260902/build-cuda/bin/llama-batched-bench
M=/data/codex-qwen-complete-20260902/models/Qwen3.5-35B-A3B-Q8_0.gguf
OUT=/data/wangkaiqi/o5cap/e3; mkdir -p $OUT
run() { local name=$1; local devs=$2; shift 2
  echo "[$(date +%T)] start $name devs=$devs :: $*" | tee -a $OUT/progress.log
  CUDA_VISIBLE_DEVICES=$devs $BB -m $M -ngl 99 -fa on --verbose "$@" > $OUT/$name.log 2>&1
  echo "[$(date +%T)] end $name rc=$?" | tee -a $OUT/progress.log
}
# n_seq_max is taken from max(-npl); -c is TOTAL ctx = n_seq_max * per-seq ctx
run s256_ctx8k_f16        2 -c 2097152 -npp 16 -ntg 1 -npl 256
run s256_ctx8k_q8         2 -c 2097152 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 256
run s256_ctx8k_q8_ub128   2 -c 2097152 -ctk q8_0 -ctv q8_0 -ub 128 -b 2048 -npp 16 -ntg 1 -npl 256
run s256_ctx16k_q8        2 -c 4194304 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 256
run s128_ctx32k_q8        2 -c 4194304 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 128
run split4_s256_ctx8k_q8  4,5,6,7 -sm layer -c 2097152 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 256
run split2_s256_ctx8k_q8  4,5 -sm layer -c 2097152 -ctk q8_0 -ctv q8_0 -npp 16 -ntg 1 -npl 256
# timing at realistic per-seq ctx 8k with q8 KV, decode-heavy
run s_tg_ctx8k_q8_pp100   2 -c 2097152 -ctk q8_0 -ctv q8_0 -npp 100 -ntg 16 -npl 1,32,128,256
echo "[$(date +%T)] ALL DONE" | tee -a $OUT/progress.log
