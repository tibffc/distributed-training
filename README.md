- Модель: EleutherAI/pythia-160m
- Датасет: wikitext-103-v1
- Метрика качества: validation perplexity (PPL)
- `seed=42`, `seq_len=512`, `global_batch_size=131072` токенов на шаг, 1 эпоха

# Инструкции к запуску

## Окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 datasets tqdm
```

---

## 1. Single-GPU baseline. Обучение модели в трёх конфигах
```bash
# fp32
CUDA_VISIBLE_DEVICES=1 nohup env python3 train_single.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype fp32 -b 32 --grad-accum 8 -s 512 --num-epochs 1 \
    > logs/part1_fp32.log 2>&1 &

# bf16
CUDA_VISIBLE_DEVICES=1 nohup env python3 train_single.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 -b 32 --grad-accum 8 -s 512 --num-epochs 1 \
    > logs/part1_bf16.log 2>&1 &

# bf16 + activation checkpointing
CUDA_VISIBLE_DEVICES=1 nohup env python3 train_single.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 --activation-checkpointing -b 32 --grad-accum 8 -s 512 --num-epochs 1 \
    > logs/part1_bf16_ckpt.log 2>&1 &
```

---

## 2. FSDP стратегии. Обучение модели с помощью `FSDP` (API `FSDP2`, NO_SHARD реализовано через DDP) на 4 GPU с тремя разными `ShardingStrategy`. В bf16, без activation checkpointing.

```bash
# NO_SHARD
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy NO_SHARD -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part2_no_shard.log

# SHARD_GRAD_OP
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy SHARD_GRAD_OP -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part2_shard_grad_op.log

# FULL_SHARD
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part2_full_shard.log
```

---

## 3. CPU Offload. Обучение модели на 2 GPU с тремя конфигами (bf16, FULL_SHARD)

```bash
# FULL_SHARD
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 2 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD -b 128 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part3_full_shard.log

# FULL_SHARD + CPUOffload
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 2 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD --cpu-offload -b 128 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part3_cpu_offload.log

# FULL_SHARD + CPUOffload + activation checkpointing
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 2 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD --cpu-offload --activation-checkpointing \
    -b 128 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part3_cpu_offload_ckpt.log
```

---

## 4. Масштабирование по числу GPU. FSDP на 1, 2 и 4 GPU с FULL_SHARD + bf16

```bash
# 1 GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 1 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD -b 64 --grad-accum 4 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part4_1gpu.log

# 2 GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 2 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD -b 64 --grad-accum 2 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part4_2gpu.log

# 4 GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --strategy FULL_SHARD -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/part4_4gpu.log
```

---

## Влияние размера модели на overhead шардирования

```bash
# NO_SHARD
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-410m \
    --strategy NO_SHARD -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/bonus_410m_no_shard.log

# SHARD_GRAD_OP
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-410m \
    --strategy SHARD_GRAD_OP -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/bonus_410m_shard_grad_op.log

# FULL_SHARD
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node 4 train_fsdp.py \
    -d Salesforce/wikitext --dataset-subset wikitext-103-v1 \
    -m EleutherAI/pythia-410m \
    --strategy FULL_SHARD -b 64 -s 512 --num-epochs 1 \
    2>&1 | tee logs/bonus_410m_full_shard.log
```
