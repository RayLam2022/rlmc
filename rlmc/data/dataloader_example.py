import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
    ]
)

transform_val = transforms.Compose([transforms.ToTensor()])


class DatasetExample(Dataset):
    def __init__(self, data, img_transform):
        self.data = torch.from_numpy(data)
        self.transform = img_transform

    def __len__(
        self,
    ):
        return self.data.shape[0]

    def __getitem__(self, item):

        # img = Image.open(file_i)
        # tensor = self.transform(img)
        tensor = self.data[item][0]
        label = self.data[item][1]
        return tensor, label


if __name__ == "__main__":

    sample_x = np.linspace(0, 1, 100)
    sample_y = np.linspace(0, 1, 100) + 1
    data = np.concatenate(
        (sample_x[np.newaxis, :].T, sample_y[np.newaxis, :].T), axis=-1
    )
    np.random.shuffle(data)
    print(data[:10])

    trainset = DatasetExample(data[:60], transform_train)
    valset = DatasetExample(data[60:], transform_val)

    trainloader = DataLoader(trainset, batch_size=10, shuffle=True)
    valloader = DataLoader(valset, batch_size=10, shuffle=False)

    for features, labels in trainloader:
        print(features)
        print(labels)
        print(features.shape)
        print(labels.shape)
        break
