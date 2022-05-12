import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

classes = ['Dark Malty Beers',
           'Fruit Beer',
           'IPA',
           'Light Beers',
           'NOT APPLICABLE',
           'Stouts']

idx_to_class = {i: j for i, j in enumerate(classes)}
class_to_idx = {value: key for key, value in idx_to_class.items()}

class Brewskis(Dataset):
  def __init__(self, image_paths, transform=False):
    self.image_paths = image_paths
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_filepath = self.image_paths[idx]
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label = image_filepath.split('/')[-2]
    label = class_to_idx[label]
    if self.transform is not None:
      image = self.transform(image=image)["image"]

    return image, label


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(59536, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

trainloader = torch.load(
    '/content/cds-5950-capstone/LoadData/train_loader.pt')

for epoch in range(10):  # loop over the dataset multiple times

  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
      running_loss = 0.0

print('Finished Training')

pytorch_total_params = sum(p.numel() for p in net.parameters())
print("Params:", pytorch_total_params)

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(
        target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


testloader = torch.load("/content/cds-5950-capstone/LoadData/test_loader.pt")

test_acc = 0.0
for i, (images, labels) in enumerate(testloader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    test_acc += get_accuracy(outputs, labels, 128)
        
print('Test Accuracy: %.2f'%( test_acc/i))

torch.save(net, '/content/cds-5950-capstone/Models/convolution.pt')
