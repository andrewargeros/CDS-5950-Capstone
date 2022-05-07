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
           'nan',
           'NOT APPLICABLE',
           'Stouts']

n_input_channels = 3
n_units_1 = 16
n_units_2 = 16
n_units_3 = 16

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

class CSEnet(nn.Module):
  def __init__(self):
    super().__init__()

    self.scs1 = SimCSE(
        in_channels=n_input_channels,
        out_channels=n_units_1,
        kernel_size=5,
        padding=0)
    self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

    self.scs2 = SimCSE(
        in_channels=n_units_1,
        out_channels=n_units_2,
        kernel_size=5,
        padding=1)
    self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

    self.scs3 = SimCSE(
        in_channels=n_units_2,
        out_channels=n_units_3,
        kernel_size=5,
        padding=1)
    self.pool3 = MaxAbsPool2d(kernel_size=4, stride=4, ceil_mode=True)
    self.out = nn.Linear(in_features=n_units_3, out_features=len(classes))

  def n_params(self):
    n = 0
    for scs in [self.scs1, self.scs2, self.scs3]:
      n += (
          np.prod(scs.weight.shape) +
          np.prod(scs.p.shape) +
          np.prod(scs.q.shape))
    n += np.prod(self.out.weight.shape)
    return n

  def forward(self, t):
    t = self.scs1(t)
    t = self.pool1(t)

    t = self.scs2(t)
    t = self.pool2(t)

    t = self.scs3(t)
    t = self.pool3(t)

    t = t.reshape(-1, n_units_3)
    t = self.out(t)

    return t

net = CSEnet()

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

torch.save(net, '/content/cds-5950-capstone/Models/SimCSE.pt')