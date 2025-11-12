import torch
from train_single_gpu import TinyCNN, ToyVision   # reuse same model + dataset class

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) load model
model = TinyCNN(num_classes=10).to(device)
state = torch.load("ckpts/ddp_tinycnn.pt", map_location=device)
model.load_state_dict(state["model"])
model.eval()

# 2) one sample (index 0)
ds = ToyVision(n=16_000, seed=42)
img, label = ds[0]
img = img.unsqueeze(0).to(device)   # add batch dimension

# 3) forward pass
with torch.no_grad():
    logits = model(img)
    pred = logits.argmax(dim=1).item()

correct = 0
for i in range(100):   # test first 100 samples
    x, y = ds[i]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    if pred == y:
        correct += 1

print("accuracy:", correct, "/ 100 =", correct/100)
