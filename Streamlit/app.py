import streamlit as st 
import numpy as np
import cv2
from typing import Optional
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

class SimCSE(nn.Conv2d):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int=3,
    padding: int=0,
    stride: int=1,
    groups: int=1,
    shared_weights: bool = False,
    log_p_init: float=.7,
    log_q_init: float=1.,
    log_p_scale: float=5.,
    log_q_scale: float=.3,
    alpha: Optional[float]=10,
    alpha_autoinit: bool=False,
    eps: float=1e-6,
):
    assert groups == 1 or groups == in_channels, " ".join([
        "'groups' needs to be 1 or 'in_channels' ",
        f"({in_channels})."])
    assert out_channels % groups == 0, " ".join([
        "The number of",
        "output channels needs to be a multiple of the number",
        "of groups.\nHere there are",
        f"{out_channels} output channels and {groups}",
        "groups."])

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    self.groups = groups
    self.shared_weights = shared_weights

    if self.groups == 1:
      self.shared_weights = False

    super(SimCSE, self).__init__(
        self.in_channels,
        self.out_channels,
        kernel_size,
        bias=False,
        padding=padding,
        stride=stride,
        groups=self.groups)

    self.kernel_size = kernel_size
    self.channels_per_kernel = self.in_channels // self.groups
    weights_per_kernel = self.channels_per_kernel * self.kernel_size ** 2
    if self.shared_weights:
      self.n_kernels = self.out_channels // self.groups
    else:
      self.n_kernels = self.out_channels
    initialization_scale = (3 / weights_per_kernel) ** .5
    scaled_weight = np.random.uniform(
        low=-initialization_scale,
        high=initialization_scale,
        size=(
            self.n_kernels,
            self.channels_per_kernel,
            self.kernel_size,
            self.kernel_size)
    )
    self.weight = torch.nn.Parameter(torch.Tensor(scaled_weight))

    self.log_p_scale = log_p_scale
    self.log_q_scale = log_q_scale
    self.p = torch.nn.Parameter(torch.full(
        (1, self.n_kernels, 1, 1),
        float(log_p_init * self.log_p_scale)))
    self.q = torch.nn.Parameter(torch.full(
        (1, 1, 1, 1), float(log_q_init * self.log_q_scale)))
    self.eps = eps

    if alpha is not None:
      self.alpha = torch.nn.Parameter(torch.full(
          (1, 1, 1, 1), float(alpha)))
    else:
      self.alpha = None
    if alpha_autoinit and (alpha is not None):
      self.LSUV_like_init()

  def LSUV_like_init(self):
    BS, CH = 32, int(self.weight.shape[1]*self.groups)
    H, W = self.weight.shape[2], self.weight.shape[3]
    device = self.weight.device
    inp = torch.rand(BS, CH, H, W, device=device)
    with torch.no_grad():
        out = self.forward(inp)
        coef = (out.std(dim=(0, 2, 3)) + self.eps).mean()
        self.alpha.data *= 1.0 / coef.view_as(self.alpha)
    return

  def forward(self, inp: torch.Tensor) -> torch.Tensor:
    p = torch.exp(self.p / self.log_p_scale)
    q = torch.exp(-self.q / self.log_q_scale)

    if self.shared_weights:
        weight = torch.tile(self.weight, (self.groups, 1, 1, 1))
        p = torch.tile(p, (1, self.groups, 1, 1))
    else:
        weight = self.weight

    return self.scs(inp, weight, p, q)

  def scs(self, inp, weight, p, q):

    weight = weight / self.weight_norm(weight)

    cos_sim = F.conv2d(
        inp,
        weight,
        stride=self.stride,
        padding=self.padding,
        groups=self.groups,
    ) / self.input_norm(inp, q)

    # Raise the result to the power p, keeping the sign of the original.
    out = cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

    # Apply learned scale parameter
    if self.alpha is not None:
      out = self.alpha.view(1, -1, 1, 1) * out
    return out

  def weight_norm(self, weight):
    # Find the l2-norm of the weights in each kernel.
    return weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()

  def input_norm(self, inp, q):
    xnorm = F.conv2d(
        inp.square(),
        torch.ones((
            self.groups,
            self.channels_per_kernel,
            self.kernel_size,
            self.kernel_size)),
        stride=self.stride,
        padding=self.padding,
        groups=self.groups)

    # Add in the q parameter. 
    xnorm = (xnorm + self.eps).sqrt() + q
    outputs_per_group = self.out_channels // self.groups
    return torch.repeat_interleave(xnorm, outputs_per_group, axis=1)


class AbsPool(nn.Module):
  def __init__(self, pooling_module=None, *args, **kwargs):
    super(AbsPool, self).__init__()
    self.pooling_layer = pooling_module(*args, **kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    pos_pool = self.pooling_layer(x)
    neg_pool = self.pooling_layer(-x)
    abs_pool = torch.where(pos_pool >= neg_pool, pos_pool, -neg_pool)
    return abs_pool

MaxAbsPool2d = partial(AbsPool, nn.MaxPool2d)

class Network(nn.Module):
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
    self.out = nn.Linear(in_features=3600, out_features=len(classes))

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

    t = t.view(t.size(0), -1)
    t = self.out(t)

    return t

net = torch.load('/Users/andrewargeros/Documents/CDS-5950-Capstone/Models/convolution.pt',
    map_location=torch.device('cpu'))

scs = torch.load('/Users/andrewargeros/Documents/CDS-5950-Capstone/Models/SimCSE.pt')

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

classes = ['Dark Malty Beers',
           'Fruit Beer',
           'IPA',
           'Light Beers',
           'nan',
           'NOT APPLICABLE',
           'Stouts']

def make_prediction(img, classes, model):
  t = test_transforms(image=img)
  b = torch.unsqueeze(t['image'], 0)
  model.eval()
  out = model(b)
  prob = F.softmax(out, dim=1)[0] * 100
  _, indices = torch.sort(out, descending=True)
  return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

st.markdown("""<style>
.css-fk4es0 {
    position: absolute;
    top: 0px;
    right: 0px;
    left: 0px;
    height: 0.75rem;
    background-image: linear-gradient(
90deg, #bc7012 0%,#efd002 100%);
    z-index: 1000020;
}
</style>""", unsafe_allow_html=True)

with st.container():
  st.title("What Beer is this?")
  st.header("Computational Data Science Capstone Project")
  st.markdown("---")
  st.markdown("#### Take a picture to guess the beer:")

  picture = st.camera_input("")

  if picture:
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    c1, c2 = st.columns(2)

    with c1:
      st.header("Convolution")
      p1 = make_prediction(img = cv2_img, classes=classes, model = net)
      st.write(p1)

    with c2:
      st.header("Sharpened Cosine Similarity")
      p2 = make_prediction(img = cv2_img, classes=classes, model = scs)
      st.write(p2)


