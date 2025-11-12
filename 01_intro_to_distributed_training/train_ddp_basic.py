"""
Basic PyTorch DDP (single node).
Key differences from single-GPU:
- Initialize process group (dist.init_process_group)
- One process per GPU launched by `torchrun`
- Set device per process via LOCAL_RANK
- Use DistributedSampler (no shuffle=True in DataLoader)
- Wrap model with DistributedDataParallel
- sampler.set_epoch(epoch) for proper shuffling each epoch
- Only rank 0 saves checkpoints / prints key logs
Run with e.g.:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp_basic.py
"""
import os, time, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class ToyVision(Dataset):
    """
    Simple synthetic dataset for quick training/debugging.

    - Generates `n` random images (3 x 64 x 64) in memory
    - Each image has a random label 0 - 9
    - Useful because:
        * No need to download any dataset
        * Always same data if you fix the random seed
        * Perfect for testing DDP logic without I/O overhead
    """
    def __init__(self, n=16_000, seed=0):
        g = torch.Generator().manual_seed(seed)
        # random images
        self.x = torch.randn(n, 3, 64, 64, generator=g)
        # random labels (10 classes)
        self.y = torch.randint(0, 10, (n,), generator=g)

    # number of samples in the dataset
    def __len__(self): return self.x.size(0)

    # return one (image, label) pair
    def __getitem__(self, i): return self.x[i], self.y[i]


# ----- same tiny model
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

def is_main_process():
    """
    Check if this is the main process (rank 0).
    Returns:
        bool: True if main process, False otherwise.
    """
    # Only rank 0 should print / save checkpoints
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_ddp():
    """
    Initialize distributed training (single-node, multi-GPU).

    What this function does:
    1) Create / join the process group so all processes can communicate
       (NCCL is the backend used for GPU → GPU communication).
    2) Read LOCAL_RANK from environment (set automatically by torchrun)
       so we know which GPU this process should use.
    3) Set the current CUDA device to that GPU, so all .to(device) operations
       and model computations happen on the correct device.

    Returns:
        local_rank (int): GPU index for this process on this machine.
                          e.g. 0, 1, 2, ...
    """
    # 1) init process group
    dist.init_process_group(backend="nccl")
    # 2) map this process to its GPU (LOCAL_RANK is set by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """
    Clean up distributed training resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Allow PyTorch to auto-pick the best conv algorithm → faster training for fixed-size inputs.
    torch.backends.cudnn.benchmark = True

    # NOTE: DDP uses DistributedSampler so each process gets a unique shard
    train_ds = ToyVision(n=16_000, seed=42)
    # Create DistributedSampler, splits dataset accross multiple GPUs in DDP training
    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    # DataLoader: Feed minibatches of data to the model during training
    train_loader = DataLoader(
        train_ds,             # the Dataset object (provides __len__ and __getitem__)
        batch_size=128,       # how many samples per mini-batch (per GPU/process)
        sampler=sampler,      # DistributedSampler splits dataset across GPUs
                              # (so do NOT also set shuffle=True)
        num_workers=4,        # number of background data-loading worker threads
        pin_memory=True,      # speed: move batch to GPU faster via pinned host memory
        persistent_workers=True  # keep workers alive between epochs (faster)
    )

    # Wrap model with DDP
    model = TinyCNN().to(device)
    model = DDP(
        model,                     # the model we want to train in parallel
        device_ids=[device.index], # which GPU this *process* should use
                                # (1 process → 1 GPU → device.index is that GPU id)
        output_device=device.index # where to place the output tensors (same GPU)
    )

    # Use cross-entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(
        model.parameters(),  # parameters of the model to update
        lr=0.1,              # learning rate (how big each parameter update step is)
        momentum=0.9         # momentum helps smooth the gradient + faster convergence
    )


    epochs = 3
    # world size (number of processes / GPUs)
    world_size = dist.get_world_size()
    if is_main_process():
        print(f"[DDP] world_size={world_size} | per-GPU batch=128 | global batch={128*world_size}")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # IMPORTANT: distinct shuffles across epochs
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()

        # Optional: sync before printing
        dist.barrier()
        if is_main_process():
            print(f"[DDP] Epoch {epoch+1}/{epochs} | loss ~ {running/len(train_loader):.3f} | {time.time()-t0:.2f}s")

    # Save only once
    if is_main_process():
        os.makedirs("ckpts", exist_ok=True)
        torch.save({"model": model.module.state_dict()}, "ckpts/ddp_tinycnn.pt")
        print("Saved ckpts/ddp_tinycnn.pt")

    cleanup_ddp()

if __name__ == "__main__":
    main()
