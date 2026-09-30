import torch, torchvision
import sys
import pathlib
from torchvision import transforms
from torch import nn, optim


transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./', train=True,
                                      transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28*28, 512)
        self.output = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = Net()
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)
print(loss)
loss.backward()

try:
    learning_rate = float(input("Learning rate: "))
except ValueError:
    sys.exit("Error, invalid learning rate, shutting down")

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epochs = input("Number of epochs: ")
while not epochs.isdigit():
        epochs = input("Error, epoch amount is a non positive integer value, try again: ")

epochs = int(epochs)

for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss/len(trainloader)}")


save_path = input("Enter path to save model: ")
path = pathlib.Path(save_path)

while not path.exists():
    save_path = input("Error, invalid path try again: ")
    path = pathlib.Path(save_path)

torch.save(model.state_dict(), save_path)
