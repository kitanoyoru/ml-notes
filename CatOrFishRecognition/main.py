import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile

from pathlib import Path
from typing import Union


class CatOrFish(nn.Module):
    def __init__(self):
        super(CatOrFish, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def is_image(path: Union[str, Path]) -> bool:
    try:
        _ = Image.open(path)
        return True
    except:
        return False


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)

        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            ouput = model(inputs)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        valid_loss /= len(val_loader.dataset)

        print(
            f"Epoch: {epoch}, Training Loss: {training_loss}, Validation loss: {valid_loss}, Accuracy: {num_correct / num_examples}"
        )


if __name__ == "__main__":
    model = CatOrFish()

    img_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tran_data_path = "./train"
    train_data = torchvision.datasets.ImageFolder(
        root=tran_data_path, transforms=img_transforms, is_valid_file=is_image
    )

    val_data_path = "./val/"
    val_data = torchvision.datasets.ImageFolder(
        root=val_data_path, transform=img_transforms, is_valid_file=check_image
    )

    test_data_path = "./test/"
    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path, transform=img_transforms, is_valid_file=check_image
    )

    batch_size = 64

    train_data_loader = torch.utils.data.Dataloader(train_data, batch_size=batch_size)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    train(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_data_loader,
        val_data_loader,
        epochs=5,
        device=device,
    )
