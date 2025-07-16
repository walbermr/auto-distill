import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3, 
            out_channels = 6, 
            kernel_size = 5
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels = 6, 
            out_channels = 16,
            kernel_size = 5
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Cifar10Trainer:
    def __init__(self):
        self.net = Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.classes = (
            "plane", 
            "car", 
            "bird", 
            "cat",
            "deer", 
            "dog",
            "frog", 
            "horse", 
            "ship", 
            "truck"
        )


    def train(self, trainLoader):
        print("Starting training.")

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainLoader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.to(self.device))
                loss = self.criterion(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        print("Finished Training")
        torch.save(self.net.state_dict(), "./cifar_net.pth")

    def validation(self, testLoader):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                outputs = self.net(images.to(self.device))
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels.to(self.device), predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=True,
        download=True, 
        transform=transform
    )
    trainLoader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=12
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=False,
        download=True, 
        transform=transform
    )
    testLoader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=12
    )

    trainer = Cifar10Trainer()
    trainer.train(trainLoader)
    trainer.validation(testLoader)


if __name__ == "__main__":
    main()