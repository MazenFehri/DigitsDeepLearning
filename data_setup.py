from torchvision import datasets
from torchvision import transforms

def get_datasets(root="data"):
    train_data = datasets.MNIST(root=root, train=True, transform=transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]),
      download=True)
    test_data = datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
    return train_data, test_data


from torch.utils.data import DataLoader

def create_dataloader(train_data,test_data,BATCH_SIZE =32):
    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)
    return train_loader,test_loader

