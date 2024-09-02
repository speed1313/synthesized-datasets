# synthesized-datasets

```
rye sync -f
```
<!-- pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 -->
<!-- pip install flashinfer -i "https://flashinfer.ai/whl/cu124/torch2.4" -->
<!-- pip install --no-build-isolation flash_attn vllm-flash-attn -->
<!-- git clone https://github.com/vllm-project/flash-attention.git
cd flash-attention
LD_LIBRARY_PATH="" MAX_JOBS=16 pip install --no-build-isolation . -->

```
OUTLINES_CACHE_DIR="/local2/tsukagoshi/.outlines"
python src/make_qa/swallow_mx.py

VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_9b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_70b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_13b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/swallow_mx.py

VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_70b.py

VLLM_ATTENTION_BACKEND="FLASHINFER" python src/make_qa/swallow_mx.py --dtype bf16
python src/make_qa/swallow_mx.py --dtype fp16
```

```
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py > /dev/null 2>&1 &
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py > /dev/null 2>&1 &
NCCL_P2P_DISABLE=1 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/swallow_mx.py --dtype fp16 > /dev/null 2>&1 &
```