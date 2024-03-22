
import argparse

samplers = ['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'uni', 'dpm', 'ddpm',  'lcm', 'orig']
models = ['lcm', '15', '15drm', '1p2p', '2i', '21', '21v', 'vzs', 'vpot'] # lcm, 15, 15drm are uncensored
un = ""
un = "low quality, poorly drawn, out of focus, blurry, tiled, segmented, oversaturated"
# un += ", letters, text, titles, graffiti, typography, watermarks, writings"
# un += ", human, people, man, girl, face"
# un += ", ugly, deformed, disfigured, mutated, mutilated, bad anatomy, malformed hands, extra limbs"

def unprompt(args):
    una = args.unprompt
    return un if una is None else '' if una=='no' else una if una[-1]=='.' else un + una if una[0]==',' else ', '.join([una, un])


class main_args:
    def __init__(self):
        self.in_txt = ''  # Text string or file to process
        self.pretxt = ''  # Prefix for input text
        self.postxt = ''  # Postfix for input text
        self.in_img = None  # Input image or directory with images (overrides width and height)
        self.img_ref = None  # Reference image or directory with images (overrides width and height)
        self.imgref_weight = 0.3  # Weight for the reference image(s), relative to the text prompt
        self.mask = None  # Path to input mask for inpainting mode (overrides width and height)
        self.unprompt = None  # Negative prompt to be used as a neutral [uncond] starting point
        self.out_dir = "_out"  # Output directory for generated images
        self.maindir = './models'  # Main SD models directory
        self.model = '15'  # SD model to use
        self.sampler = 'ddim'  # Sampler to use
        self.vae = 'ema'  # VAE option
        self.cfg_scale = 7.5  # Prompt guidance scale
        self.strength = 1  # Strength of image processing. 0 = preserve img, 1 = replace it completely
        self.img_scale = None  # Image guidance scale for Instruct pix2pix. None = disabled it
        self.ddim_eta = 0.  # DDIM eta
        self.steps = 50  # Number of diffusion steps
        self.batch = 1  # Batch size
        self.vae_batch = 8  # Batch size for VAE decoding
        self.num = 1  # Repeat prompts N times
        self.seed = None  # Image seed
        self.load_token = None  # Path to the text inversion embeddings file
        self.load_custom = None  # Path to the custom diffusion delta checkpoint
        self.load_lora = None  # Path to the LoRA file
        self.control_mod = None  # Path to the ControlNet model
        self.control_img = None  # Path to the ControlNet driving image (contour, pose, etc)
        self.control_scale = 0.7  # ControlNet effect scale
        self.cguide = False  # Use noise guidance for interpolation, instead of cond lerp
        self.freeu = False  # Use FreeU enhancement (Fourier representations in Unet)
        self.sag_scale = 0  # Self-attention guidance scale
        self.size = None  # Image size, multiple of 8
        self.lowmem = False  # Offload subnets onto CPU for higher resolution [slower]
        self.invert_mask = False  # Invert mask
        self.allref = False  # Apply all reference images at once or pick one by one?
        self.verbose = False  # Verbose output

# Crear una instancia de la configuración
#args = ConfiguracionDefault()

# Ejemplo de cómo acceder a un atributo
#print(configuracion.in_txt)  # Imprime: ''

