
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
    def __init__(self,a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
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


    def setseed(self, seed=None):
        self.seed = seed or int((time.time()%1)*69696)
        self.g_ = torch.Generator("cuda").manual_seed(self.seed)


    def load_model_external(self, model_path):
        SDload = StableDiffusionPipeline.from_single_file if os.path.isfile(model_path) else StableDiffusionPipeline.from_pretrained
        self.pipe = SDload(model_path, torch_dtype=torch.float16, safety_checker=None)
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer    = self.pipe.tokenizer
        self.unet         = self.pipe.unet
        self.vae          = self.pipe.vae
        self.scheduler    = self.pipe.scheduler if self.a.sampler=='orig' else self.set_scheduler(self.a)
        self.sched_kwargs = {}

    def load_model_custom(self, a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        # paths
        self.clipseg_path = os.path.join(a.maindir, 'xtra/clipseg/rd64-uni.pth')
        vtype  = a.model[-1] == 'v'
        vidtype = a.model[0] == 'v'
        self.subdir = 'v2v' if vtype else 'v2' if vidtype or a.model[0]=='2' else 'v1'

        if vtype and not isxf: # scheduler.prediction_type == "v_prediction":
            print(" V-models require xformers! install it or use another model"); exit()

        # text input
        txtenc_path = os.path.join(a.maindir, self.subdir, 'text-' + a.model[2:] if a.model[2:] in ['drm'] else 'text')
        if text_encoder is None:
            text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=torch.float16, local_files_only=True)
        if tokenizer is None:
            tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=torch.float16, local_files_only=True)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if unet is None:
            unet_path = os.path.join(a.maindir, self.subdir, 'unet' + a.model)
            if vidtype:
                from diffusers.models import UNet3DConditionModel as UNet
            else:
                from diffusers.models import UNet2DConditionModel as UNet
            unet = UNet.from_pretrained(unet_path, torch_dtype=torch.float16, local_files_only=True)
        if not isxf and isinstance(unet.config.attention_head_dim, int): unet.set_attention_slice(unet.config.attention_head_dim // 2) # 8
        self.unet = unet

        if vae is None:
            vae_path = 'vae'
            if a.model[0]=='1' and a.vae != 'orig':
                vae_path = 'vae-ft-mse' if a.vae=='mse' else 'vae-ft-ema'
            vae_path = os.path.join(a.maindir, self.subdir, vae_path)
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        if a.lowmem: vae.enable_tiling() # higher res, more free ram
        elif vidtype or isset(a, 'animdiff') or not isxf: vae.enable_slicing()
        self.vae = vae

        if scheduler is None:
            scheduler = self.set_scheduler(a, self.subdir, vtype)
        self.scheduler = scheduler

        self.pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler)



# sliding sampling for long videos
# from https://github.com/ArtVentureX/comfyui-animatediff/blob/main/animatediff/sliding_schedule.py
def ordered_halving(val, verbose=False): # Returns fraction that has denominator that is a power of 2
    bin_str = f"{val:064b}" # get binary value, padded with 0s for 64 bits
    bin_flip = bin_str[::-1] # flip binary value, padding included
    as_int = int(bin_flip, 2) # convert binary to int
    final = as_int / (1 << 64) # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616, or 1 with 64 zero's
    if verbose: print(f"$$$$ final: {final}")
    return final
# generate lists of latent indices to process
def uniform_slide(step, num_frames, ctx_size=16, ctx_stride=1, ctx_overlap=4, loop=True, verbose=False):
    if num_frames <= ctx_size:
        yield list(range(num_frames))
        return
    ctx_stride = min(ctx_stride, int(np.ceil(np.log2(num_frames / ctx_size))) + 1)
    pad = int(round(num_frames * ordered_halving(step, verbose)))
    fstop = num_frames + pad + (0 if loop else -ctx_overlap)
    for ctx_step in 1 << np.arange(ctx_stride):
        fstart = int(ordered_halving(step) * ctx_step) + pad
        fstep = ctx_size * ctx_step - ctx_overlap
        for j in range(fstart, fstop, fstep):
            yield [e % num_frames for e in range(j, j + ctx_size * ctx_step, ctx_step)]

class CrossAttnStoreProcessor: # processes and stores attention probabilities
    def __init__(self):
        self.attention_probs = None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) # linear proj
        hidden_states = attn.to_out[1](hidden_states) # dropout
        return hidden_states

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img


