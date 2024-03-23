
import os, sys
import time
import numpy as np
from contextlib import nullcontext
from einops import rearrange

import torch
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_accelerate_available, is_accelerate_version


ruta_especifica = "/content/SDFU/src/xtra"
if ruta_especifica in sys.path:
    print("La ruta ya está en sys.path")
else:
    print("La ruta no está en sys.path, añadiéndola...")
    sys.path.append(ruta_especifica)


from .text import multiprompt
from .utils import img_list, load_img, makemask, isok, isset, progbar, file_list
from .args import models, unprompt

import logging



class sdfu:
    def __init__(self,a):
        print(a.in_txt)
        print("act")









