
# 01 — From Single-GPU Training to Basic DDP

This chapter introduces the **first step** in distributed deep learning:
starting from a *normal single-GPU training script*, and then refactoring it into a
**Distributed Data Parallel (DDP)** training script that can scale to multi-GPU.

The goal is **not performance** yet — the goal is to understand structure, concepts, and why multi-GPU code is different.

---

## Files in this chapter

| file | description |
|---|---|
| `train_single_gpu.py` | normal single-GPU training loop (baseline) |
| `train_ddp_basic.py` | DDP version of the same training (2 GPUs or more) |

Both scripts use a synthetic dataset (no download needed) and a tiny CNN.

---

## Key differences (single GPU vs DDP)

| area | single-GPU | DDP |
|---|---|---|
| launch command | `python train_single_gpu.py` | `torchrun --nproc_per_node=2 train_ddp_basic.py` |
| model | plain `model.to(device)` | wrapped with `DistributedDataParallel(model, ...)` |
| dataset | `shuffle=True` in DataLoader | **DistributedSampler** (each GPU gets its shard) |
| device | 1 GPU or CPU | **one process per GPU**, using `LOCAL_RANK` |
| print / save | anywhere | only **rank 0** prints & saves checkpoint |

The architecture of the model and the optimizer are almost the same — the difference is how we launch, shard data, and wrap the model.

---

## Why this matters

Before you can do:

- multi-GPU training for speed (bigger global batch)
- FSDP / ZeRO for memory sharding
- multi-node training
- large model training

you must understand **basic single-node DDP**.

DDP is the “base layer” that everything else builds on.

---

## What you should learn from this chapter

- how to launch distributed training with `torchrun`
- what a “rank” is (each GPU = one process)
- how DistributedSampler splits dataset automatically
- why *only rank 0* should write logs & checkpoints
- the minimal code structure difference from single-GPU

If you understand these 5 points → the rest of distributed training becomes much easier.

---

## How to run

### Single-GPU version:
```bash
python train_single_gpu.py
````

### DDP (single node, 2 GPUs)

```bash
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp_basic.py
```

If you only have 1 GPU, you can still “fake-DDP” test:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_ddp_basic.py
```

This confirms that your distributed init logic is correct
**before** you pay money on RunPod / AWS.

## Keywords



**Local Rank, Global Rank**

In distributed training, every process gets an ID called a **rank**.
**Global rank** is the unique ID of a process across the entire training job (all GPUs, even across multiple machines).
**Local rank** is the process ID inside a *single machine*, which matches the GPU index on that machine.
For beginner multi-GPU training, we normally start with **all GPUs in one machine** — here local rank simply tells each process which GPU to use (GPU0, GPU1, etc.), while global rank identifies which process is the overall rank-0 “leader” for logging and checkpoint saving.

**NCCL**

NCCL (NVIDIA Collective Communications Library) is a high-performance communication library optimized for GPU-to-GPU data exchange. PyTorch uses it as the default backend for multi-GPU training because it efficiently implements collective operations like all-reduce, which are required to synchronize gradients across GPUs during Distributed Data Parallel (DDP). In short, NCCL is the fast communication engine that makes multi-GPU training scale efficiently on NVIDIA hardware.

**Cross Entropy Loss**

$J = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} y_{i,c}\;\log\left( \frac{e^{z_{i,c}}}{\sum_{k=1}^{K} e^{z_{i,k}}} \right)$

This is the most commonly used loss function for multi-class classification. It takes the model’s logits, applies softmax internally (in a numerically stable way), converts them into class probabilities, and penalizes the model when the predicted probability for the correct class is low. The result is a single scalar loss value used for backpropagation.
