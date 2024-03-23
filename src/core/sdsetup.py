
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
logging.getLogger('diffusers').setLevel(logging.ERROR)
try:
    import xformers; isxf = True
except: isxf = False
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

device = torch.device('cuda')

class SDpipe(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, image_encoder=None, \
                 safety_checker=None, feature_extractor=None, requires_safety_checker=False):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, image_encoder, None, None, requires_safety_checker=False)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        if image_encoder is not None:
            self.register_modules(image_encoder=image_encoder)



class sdfu:
    def __init__(self,a):
        print(a.in_txt)
        print("act")


        self.a = a
        self.device = device
        self.run_scope = nullcontext # torch.autocast
        if not isset(a, 'maindir'): a.maindir = './models' # for external scripts
        self.setseed(a.seed if isset(a, 'seed') else None)
        a.unprompt = unprompt(a)

        self.use_lcm = False
        if a.model == 'lcm':
            self.use_lcm = True
            a.model = os.path.join(a.maindir, 'lcm')
            self.a.sampler = 'orig'
            if self.a.cfg_scale > 3: self.a.cfg_scale = 2
            if self.a.steps > 12: self.a.steps = 4 # if steps are still set for ldm schedulers

        if a.model in models:
            self.load_model_custom(self.a, vae, text_encoder, tokenizer, unet, scheduler)
        else: # downloaded or url
            self.load_model_external(a.model)
        if not self.a.lowmem: self.pipe.to(device)
        self.use_kdiff = hasattr(self.scheduler, 'sigmas') # k-diffusion sampling

        # load finetuned stuff
        mod_tokens = None
        # if isset(a, 'load_lora') and os.path.isfile(a.load_lora): # lora
            # self.pipe.load_lora_weights(a.load_lora, low_cpu_mem_usage=True)
            # self.pipe.fuse_lora()
            # if a.verbose: print(' loaded LoRA', a.load_lora)
        if isset(a, 'load_custom') and os.path.isfile(a.load_custom): # custom diffusion
            from .finetune import load_delta, custom_diff
            self.pipe.unet = custom_diff(self.pipe.unet, train=False)
            mod_tokens = load_delta(torch.load(a.load_custom), self.pipe.unet, self.pipe.text_encoder, self.pipe.tokenizer)
        elif isset(a, 'load_token') and os.path.exists(a.load_token): # text inversion
            from .finetune import load_embeds
            emb_files = [a.load_token] if os.path.isfile(a.load_token) else file_list(a.load_token, 'pt')
            mod_tokens = []
            for emb_file in emb_files:
                mod_tokens += load_embeds(torch.load(emb_file), self.pipe.text_encoder, self.pipe.tokenizer)
        if mod_tokens is not None: print(' loaded tokens:', mod_tokens[0] if len(mod_tokens)==1 else mod_tokens)

        # load controlnet
        if isset(a, 'control_mod'):
            if not os.path.exists(a.control_mod): a.control_mod = os.path.join(a.maindir, 'control', a.control_mod)
            assert os.path.exists(a.control_mod), "Not found ControlNet model %s" % a.control_mod
            if a.verbose: print(' loading ControlNet', a.control_mod)
            from diffusers import ControlNetModel
            self.cnet = ControlNetModel.from_pretrained(a.control_mod, torch_dtype=torch.float16)
            if not self.a.lowmem: self.cnet.to(device)
            self.pipe.register_modules(controlnet=self.cnet)
        self.use_cnet = hasattr(self, 'cnet')

        # load animatediff = before ip adapter, after custom diffusion !
        if isset(a, 'animdiff'):
            if not os.path.exists(a.animdiff): a.animdiff = os.path.join(a.maindir, a.animdiff)
            assert os.path.exists(a.animdiff), "Not found AnimateDiff model %s" % a.animdiff
            if a.verbose: print(' loading AnimateDiff', a.animdiff)
            from diffusers.models import UNetMotionModel, MotionAdapter
            motion_adapter = MotionAdapter.from_pretrained(a.animdiff)
            self.unet = UNetMotionModel.from_unet2d(self.unet, motion_adapter)
            self.scheduler = self.set_scheduler(a) # k-samplers must be loaded after unet
            if not self.a.lowmem: self.unet.to(device)
            self.pipe.register_modules(unet = self.unet)

        if isset(a, 'load_lora'): # lora = after animdiff !
            self.pipe.load_lora_weights(a.load_lora, low_cpu_mem_usage=True)
            self.pipe.fuse_lora() # (lora_scale=0.7)
            if a.verbose: print(' loaded LoRA', a.load_lora)

        if isset(a, 'animdiff'): # after lora !
            self.pipe.register_modules(motion_adapter = motion_adapter)

        # load ip adapter = after animatediff
        if isset(a, 'img_ref'):
            assert '15' in a.model, "!! IP adapter models are hardcoded for SD 1.5"
            if a.verbose: print(' loading IP adapter for images', a.img_ref)
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection as CLIPimg
            self.image_preproc = CLIPImageProcessor.from_pretrained(os.path.join(a.maindir, 'image/preproc_config.json')) # openai/clip-vit-base-patch32
            self.image_encoder = CLIPimg.from_pretrained(os.path.join(a.maindir, 'image'), torch_dtype=torch.float16).to(device)
            self.unet._load_ip_adapter_weights(torch.load(os.path.join(a.maindir, 'image/ip-adapter_sd15.bin'), map_location="cpu"))
            self.pipe.register_modules(image_encoder = self.image_encoder)
            self.pipe.set_ip_adapter_scale(a.imgref_weight)

        self.final_setup(a)







