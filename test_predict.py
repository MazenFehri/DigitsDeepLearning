import torch
from torch import nn
from torchmetrics import Accuracy
from model import MNIST_v2

model = MNIST_v2(
    input_shape=1, hidden_units=120, output_shape=10,
)
model.load_state_dict(torch.load("model/mnist_v2.pth"))
model.eval()


from data_setup import get_datasets
_, test_data = get_datasets()
img, label = test_data[0]

with torch.inference_mode():
    pred = model(img.unsqueeze(0)).argmax(dim=1).item() 

assert pred == label, f"expected {label}, got {pred}"
print(f"OK: predicted {pred}")

