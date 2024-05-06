
import os
import sys

sys.path.append("/DATA_EDS2/AIGC/2312/xuhr2312/workspace/ControlNet")

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6"

from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
from datasets.cityscapes import CityscapesDataset

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents



full_data=[]
full_label=[]
full_labelIds=[]

with open("/DATA_EDS2/AIGC/2312/xuhr2312/workspace/ControlNet/datasets/cityscapes.txt", 'r') as f:
    for line in f.readlines():
        data, label = line.strip().split(',')
        full_data.append(data)
        full_label.append(label.replace('labelIds', 'color'))
        full_labelIds.append(label.replace('labelIds', 'labelTrainIds'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                                                scheduler=inverse_scheduler,
                                                safety_checker=None,
                                                torch_dtype=dtype)
pipe.to(device)
vae = pipe.vae


@torch.no_grad()
def ddim_inversion(imgname: str, num_steps: int = 50, verify: Optional[bool] = False) -> torch.Tensor:

    input_img = load_image(imgname).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    # verify
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
        image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.savefig("output.png")
        plt.show()
    return inv_latents


with open('/DATA_EDS2/AIGC/2312/xuhr2312/workspace/ControlNet/datasets/inv.txt', 'w') as f:
    # 按顺序处理每个图像，并将结果存储在文件夹中
    for i, imgname in enumerate(full_data):
        inv = ddim_inversion(imgname, num_steps=20, verify=False)
        base_name = os.path.basename(imgname)  # 获取文件名，结果是"dusseldorf_000148_000019_leftImg8bit.png"
        file_name_without_extension = os.path.splitext(base_name)[0]  # 去掉扩展名，结果是"dusseldorf_000148_000019_leftImg8bit"

        # 将结果存储为torch文件
        output_path = f'/DATA_EDS2/AIGC/2312/xuhr2312/workspace/ControlNet/data/cityscapes_inv_float32/inv_{file_name_without_extension}.pt'
        torch.save(inv, output_path)

        # 将文件的路径写入txt文件
        f.write(output_path + '\n')