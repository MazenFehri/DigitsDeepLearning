import torch
import torch.nn as nn

def make_predictions(model:nn.Module , data):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data :
            sample = torch.unsqueeze(sample,dim=0)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(),dim=0)
            pred_probs.append(pred_prob)
    
    return torch.stack(pred_probs)

from data_setup import get_datasets
_ , test_data = get_datasets(root="data")
class_names = test_data.classes

from model import MNIST_v3

model_v3 = MNIST_v3(1,120,10)
model_v3.load_state_dict(torch.load("model/mnist_v3.pth"))

import random
random.seed(424)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)



pred_probs= make_predictions(model=model_v3, 
                             data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  pred_label = class_names[pred_classes[i]]

  truth_label = class_names[test_labels[i]] 

  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);

plt.savefig("images/sample_predictions.png")
plt.show()

from data_setup import create_dataloader
train_loader,test_loader = create_dataloader(_,test_data)

y_preds =[]

model_v3.eval()
with torch.inference_mode():
    for x , y in test_loader:
        y_logits = model_v3(x)
        y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
        y_preds.append(y_pred)
y_pred_tensor = torch.cat(y_preds)

    

import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);



fig.savefig("images/confusion_matrix.png")
plt.show()