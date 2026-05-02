import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from cnn_torch import CNN

def load_datasets():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    printed = datasets.ImageFolder(
        "printed_dataset/",
        transform=transform
    )

    return printed

def train_model(device="cpu", log_callback=None, epochs=5, batch_size=64, lr=1e-3):
    device = torch.device(device)

    dataset = load_datasets()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN(num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = len(loader)

    for epoch in range(1, epochs + 1):
        model.train()
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if log_callback:
                log_callback(epoch, step, total_steps, loss.item())

    torch.save(model.state_dict(), "model.pth")

