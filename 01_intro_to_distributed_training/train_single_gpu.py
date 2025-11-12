"""
Single-process, single-GPU (or CPU) training.
Key traits:
- No torch.distributed
- No DistributedSampler
- Model is used directly (no DDP wrapper)
- Can run with:  python train_single_gpu.py
"""
import os, time, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----- Tiny synthetic dataset: images 3x64x64, 10 classes
class ToyVision(Dataset):
    """
    Simple synthetic dataset for quick training/debugging.
    - Generates `n` random images (3 x 64 x 64) in memory
    - Each image has a random label 0 - 9
    - Useful because:
        * No need to download any dataset
        * Always same data if you fix the random seed
        * Perfect for testing training logic without I/O overhead
    """
    def __init__(self, n=16_000, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 64, 64, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ----- Tiny CNN (fast to train)
class TinyCNN(nn.Module):
    """
    Tiny CNN for quick training/debugging.
    Architecture:
    - Conv2d(3,32,3) + ReLU
    - Conv2d(32,64,3) + ReLU
    - MaxPool2d(2)
    - Conv2d(64,128,3) + ReLU
    - MaxPool2d(2)
    - AdaptiveAvgPool2d((1,1))
    - Flatten
    - Linear(128, num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # 16x16
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # NOTE: regular DataLoader with shuffle=True (no DistributedSampler here)
    train_ds = ToyVision(n=16_000, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = TinyCNN().to(device)                     # NOTE: plain model (no DDP)
    criterion = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    epochs = 3
    for epoch in range(epochs):
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
        print(f"[Single] Epoch {epoch+1}/{epochs} | loss ~ {running/len(train_loader):.3f} | {time.time()-t0:.2f}s")

    os.makedirs("ckpts", exist_ok=True)
    torch.save({"model": model.state_dict()}, "ckpts/ckpt_single.pt")
    print("Saved ckpt_single.pt")

if __name__ == "__main__":
    main()
