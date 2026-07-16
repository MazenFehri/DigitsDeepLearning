
import torch
import torch.nn as nn
import torchmetrics

def train_step( model :torch.nn.Module,
                train_loader : torch.utils.data.DataLoader,
                loss_fn :torch.nn.modules.loss ,
                optimizer : torch.optim 
                ):
        train_loss = 0
        model.train()
        for batch , (X , y) in enumerate(train_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred,y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 400==0:
                print(f"looked at {(batch+1)*32} samples")
        train_loss = train_loss /len(train_loader)

        return train_loss



def test_step(model :torch.nn.Module,
            test_loader : torch.utils.data.DataLoader,
            loss_fn :torch.nn.modules.loss,
            acc_fn : torchmetrics.Accuracy 
            ):
    test_loss,test_acc = 0,0
    acc_fn.reset()
    model.eval()
    with torch.inference_mode():
        for batch,(X,y) in enumerate(test_loader):
            y_pred = model(X)
            pred_label = y_pred.argmax(dim=1)
            loss = loss_fn(y_pred,y)
            test_loss +=loss
            test_acc +=acc_fn(pred_label,y)
        test_loss =test_loss/len(test_loader)
        test_acc = test_acc/len(test_loader)
    return test_loss , test_acc


def train(model, train_loader, test_loader, loss_fn, optimizer, acc_fn, epochs):
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, acc_fn)
        print(f"epoch {epoch} | train_loss: {train_loss:.5f} | "
              f"test_loss: {test_loss:.5f} | test_acc: {test_acc*100:.2f}%")