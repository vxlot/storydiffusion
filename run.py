from diffusers.models.attention_processor import AttnProcessor2_0
import numpy as np
import torch
import random
import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
import os

# Search keywords and modify using Ctrl+F:
# id_length 一次生成几张图
# sd_model_path 选择sd模型
# hub_dir 从本地加载需要指定cache_dir
# style_name 风格选择

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Japanese Anime",
        "prompt": "anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution",
        "negative_prompt": "low quality, low resolution"
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Disney Charactor",
        "prompt": "A Pixar animation character of {prompt} . pixar-style, studio anime, Disney, high-quality",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Comic book",
        "prompt": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    }
]
# 学到了，存储数据是一种格式，提取数据可以变成另一种格式
styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

class ConsistentAttnProcessor2_0(torch.nn.Module):

    def __init__(self, hidden_size = None, cross_attention_dim=None, id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):

        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64, height, width
        # skip in early step [attention_mask=None]
        if cur_step <5:
            hidden_states = self.__call_norm__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:   # 256 1024 4096
            random_number = random.random()
            rand_num = 0.3 if cur_step <20 else 0.1
            if random_number > rand_num:
                if hidden_states.shape[1] == (height//32) * (width//32):  # 每张图的特征数
                    attention_mask = mask1024
                else:
                    attention_mask = mask4096
                hidden_states = self.__call_batch_attn__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call_norm__(attn, hidden_states, None, attention_mask, temb)
        attn_count +=1
        if attn_count == total_count:  # 36
            attn_count = 0
            cur_step += 1  
            mask1024, mask4096 = cal_attn_mask_xl(self.id_length, sa32, sa64, height, width, 
                                                  device=self.device, dtype= self.dtype)
        return hidden_states
    
    def __call_batch_attn__(self, attn, hidden_states, encoder_hidden_states=None,
                  attention_mask=None, temb=None):
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        
        total_batch_size, nums_token, channel = (hidden_states.shape)
        img_nums = total_batch_size//2 
        # [8, 576, 1280]->[2, 4, 576, 1280]->[2, 2304, 1280]
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = (hidden_states.shape)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # attn_mask:[2304, 2304] 
        # qkv:[batch,attn.heads,sequence_len,head_dim][2, 20, 2304, 64]
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        # 变回 [8, 576, 1280]
        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
       
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    
    def __call_norm__(self, attn, hidden_states, encoder_hidden_states=None,
                  attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_attn_mask_xl(id_length, sa32, sa64, height, width, device="cuda", dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)  # 576
    nums_4096 = (height // 16) * (width // 16)  # 2304
    # (1, id_length * nums_1024)
    bool_matrix1024 = torch.rand((1, id_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, id_length * nums_4096),device = device,dtype = dtype) < sa64
    # (id_length, id_length * nums_1024)
    bool_matrix1024 = bool_matrix1024.repeat(id_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(id_length,1)
    for i in range(id_length):
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    # (id_length * nums_1024, id_length * nums_1024)
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,id_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,id_length * nums_4096)
    return mask1024,mask4096



DEFAULT_STYLE_NAME = "(No style)"
global models_dict
models_dict = {
   "RealVision":"SG161222/RealVisXL_V4.0" ,
   "SDXL":"stabilityai/stable-diffusion-xl-base-1.0" ,
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}
global attn_count, total_count, id_length, cur_step
global sa32, sa64, height,width
global attn_procs, unet
attn_count, total_count, cur_step, id_length = 0, 0, 0, 5
sa32, sa64, height, width = 0.5, 0.5, 768, 768

device="cuda"
attn_procs_dict = {}
guidance_scale = 5.0
seed = 2047
num_steps = 50

sd_model_path = models_dict["Unstable"] 
hub_dir = "/root/autodl-tmp/hub"  # load locally
pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, 
                                                 cache_dir=hub_dir, 
                                                 torch_dtype=torch.float16, 
                                                 use_safetensors=False)
pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
unet = pipe.unet

# Insert PairedAttention 
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if cross_attention_dim is None and (name.startswith("up_blocks") ) :
        attn_procs_dict[name] =  ConsistentAttnProcessor2_0(id_length = id_length)
        total_count +=1
    else:
        attn_procs_dict[name] = AttnProcessor2_0()
print("successsfully load consistent self-attention")
print(f"number of spatial processor : {total_count}")
unet.set_attn_processor(attn_procs_dict)

global mask1024,mask4096
mask1024, mask4096 = cal_attn_mask_xl(id_length,
                                      sa32,
                                      sa64,
                                      height,
                                      width,
                                      device=device,
                                      dtype= torch.float16)

general_prompt = "a girl with white shirt and short hair"
negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
prompt_array = ["at cafe, read menu",
                "order a cup of coffee",
                "drink coffee",
                "She put the phone to her ear and called her friend",
                "stand and prepare to go",
                "go out of the cafe"]

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

# Set the generated Style
style_name = "Cinematic"
setup_seed(seed)
generator = torch.Generator(device="cuda").manual_seed(seed)
prompts = [general_prompt + "," + prompt for prompt in prompt_array]
id_prompts = prompts[:id_length]

torch.cuda.empty_cache()

# inference！
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
id_images = pipe(id_prompts, 
                 num_inference_steps=num_steps, 
                 guidance_scale=guidance_scale,  
                 height=height, 
                 width=width, 
                 negative_prompt=negative_prompt,
                 generator=generator).images

for id, id_image in enumerate(id_images):
    if id == 0:
        os.makedirs("out", exist_ok=True)
    id_image.save(f"out/{id}.png")





