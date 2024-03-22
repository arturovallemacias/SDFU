
import os
import sys

import torch


print(sys.path)


ruta_especifica = "/content/SDFU/src/core"
if ruta_especifica in sys.path:
    print("La ruta ya está en sys.path")
else:
    print("La ruta no está en sys.path, añadiéndola...")
    sys.path.append(ruta_especifica)

#from core.sdsetup import estaclase
from core.args import args
#, samplers
from core.text import read_txt, multiprompt
from core.utils import load_img, save_img, calc_size, isok, isset, img_list, basename, progbar, save_cfg

 

@torch.no_grad()
def main():
    a = args()
    print(a.in_txt)
    sd = SDfu(a)

if __name__ == '__main__':
    main()

