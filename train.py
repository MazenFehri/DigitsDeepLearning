from data_setup import get_datasets , create_dataloader
from engine import train

from model import MNIST_v3

import torch.nn as nn
import torch
import torchmetrics


BATCH_SIZE = 32
HIDDEN_UNITS = 120
EPOCHS = 5
LR = 0.1
SAVE_PATH = "model/mnist_v3.pth"

if __name__ == "__main__":
    torch.manual_seed(42)

    train_data , test_data = get_datasets(root="data")
    train_loader,test_loader = create_dataloader(train_data=train_data,
                                                 test_data=test_data,
                                                 BATCH_SIZE=BATCH_SIZE,
                                                 )

    model_v3 =MNIST_v3(
        input_shape=1,
        hidden_units=HIDDEN_UNITS,
        output_shape=10
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model_v3.parameters(), lr=LR)
    acc_fn = torchmetrics.Accuracy(task = 'multiclass',num_classes =10)


    train(
        model=model_v3,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        acc_fn=acc_fn,
        epochs=EPOCHS
    )

    import os
    os.makedirs("model", exist_ok=True)
    torch.save(model_v3.state_dict(), SAVE_PATH)
    print(f"saved to {SAVE_PATH}")