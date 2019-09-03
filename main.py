import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import seaborn as sb
import PIL
from PIL import Image
import argparse
from collections import OrderedDict


classifier = nn.Sequnetial(OrderedDict([
    ('dropout', nn.Dropout(dropout)),
    ('inputs', nn.Linear
]))