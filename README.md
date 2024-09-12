# synthesized-datasets

```
rye sync -f
pip install --no-build-isolation flash_attn vllm-flash-attn
```
<!-- pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 -->
<!-- pip install flashinfer -i "https://flashinfer.ai/whl/cu124/torch2.4" -->
<!-- git clone https://github.com/vllm-project/flash-attention.git
cd flash-attention
LD_LIBRARY_PATH="" MAX_JOBS=16 pip install --no-build-isolation . -->
<!-- pip install --force-reinstall --no-build-isolation flash_attn vllm-flash-attn -->

```
OUTLINES_CACHE_DIR="/local2/tsukagoshi/.outlines"
python src/make_qa/swallow_mx.py

VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_9b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_70b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_13b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/swallow_mx.py

VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_mini.py
VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_moe.py

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_9b.py --tp 1

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py --tp 1
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py --tp 1
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py --tp 1

CUDA_VISIBLE_DEVICES=1,2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/gemma2_27b.py --tp 2

VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_70b.py

VLLM_ATTENTION_BACKEND="FLASHINFER" python src/make_qa/swallow_mx.py --dtype bf16
VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/swallow_mx.py --dtype fp16
VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_13b.py --dtype fp16
VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/sarashina2_7b.py --dtype fp16

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_mini.py --dtype bf16 --tp 1

VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_mini.py --dtype fp16

CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 2

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 1
```

```
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py > /dev/null 2>&1 &
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py > /dev/null 2>&1 &
VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/swallow_mx.py --dtype fp16 > /dev/null 2>&1 &
VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/sarashina2_7b.py --dtype fp16 > /dev/null 2>&1 &


CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_9b.py --tp 1 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype bf16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype bf16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype bf16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype bf16 --tp 1 > /dev/null 2>&1 &


CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/gemma2_27b.py --tp 1 > /dev/null 2>&1 &

VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 2 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/phi3_5_mini.py --dtype fp16 --tp 1 > /dev/null 2>&1 &


VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_qa/sarashina2_70b.py  > /dev/null 2>&1 &
```


```
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_paraphrase/gemma2_27b.py
VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b.py > /dev/null 2>&1 &


CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_paraphrase/gemma2_9b.py --tp 1
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_paraphrase/gemma2_27b.py --tp 1
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork python src/make_paraphrase2/gemma2_27b.py --tp 1

<!-- 温かみのある一行ごと実行をしないとtimestampがズレなくて同じ事例をもとに同じ生成結果が出てしまう -->
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b.py --tp 1 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_9b.py --tp 1 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b_balance.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b_balance.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b_balance.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase/gemma2_27b_balance.py --tp 1 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_9b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_9b.py --tp 1 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND="FLASHINFER" VLLM_WORKER_MULTIPROC_METHOD=fork nohup python src/make_paraphrase2/gemma2_27b.py --tp 1 > /dev/null 2>&1 &
```