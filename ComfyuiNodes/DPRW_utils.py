import torch
import os
import sys
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from scipy.stats import norm
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import logging
from datetime import datetime
from scipy.stats import norm, kstest
import math
import cv2
from PIL import Image
from skimage.metrics import structural_similarity
# import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import io
import os
from tqdm import tqdm
from typing import Union, Tuple, Optional
import json
import cbor2
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

MAX_RESOLUTION = 8192

def set_random_seed(seed: int) -> None:
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def choose_watermark_length(total_blocks_needed: int) -> int:
    """根据可用块数选择水印长度"""
    if total_blocks_needed >= 256 * 32:
        return 256
    elif total_blocks_needed >= 128 * 32:
        return 128
    elif total_blocks_needed >= 64 * 32:
        return 64
    return 32

def validate_hex(hex_str: str, expected_length: int, default: bytes) -> bytes:
    """验证并生成十六进制字符串，若无效则返回默认值"""
    if hex_str and len(hex_str) == expected_length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
        return bytes.fromhex(hex_str)
    return default

def common_ksampler(model, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, latent,
                    denoise: float = 1.0, disable_noise: bool = False, start_step = None, last_step= None,
                    force_full_denoise: bool = False, use_dprw: bool = False, watermarked_latent_noise=None):
    """
    通用的 KSampler 函数，处理潜在表示采样并支持 DPRW 水印噪声
    """
    latent_image = latent["samples"]
    # 如果使用DPRW，这里的latent_image是first_stage的latent
    # 而noise则是watermarked_latent_noise
    if latent_image.shape[1] == 4:
        # 调用这个函数确保能适用于flux
        # 如果latent_image的shape是(1,4,x,x)，则需要进行fix，它会将(1,4,x,x)的噪声，转换为(1,16,x,x)的噪声（但由于传入进来的是empty_latent,所以直接将维度扩大即可
        # 当如果是(1,16,x,x)），于是就可以不处理了，不过这代码可以保留，它不会影响
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if use_dprw and watermarked_latent_noise is not None:
        noise = watermarked_latent_noise["samples"]
    elif disable_noise:
        noise = torch.zeros_like(latent_image, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    init_noise = {}
    init_noise["samples"] = noise.clone()
    noise_mask = latent.get("noise_mask", None)
    # print(f"noise_mask: {noise_mask} ,{type(noise_mask)}")
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out,init_noise)

def Gaussian_test(noise: torch.Tensor,logger) -> bool:
        if isinstance(noise, torch.Tensor):
            noise = noise.cpu().numpy()
        samples = noise.flatten()
        _, p_value = kstest(samples, 'norm', args=(0, 1))
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Restored noise contains NaN or Inf values")
        if np.var(samples) == 0:
            raise ValueError("Restored noise variance is 0")
        if p_value < 0.05:
            raise ValueError(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        logger.info(f"Gaussian test passed: p={p_value:.4f}")
        return True


# 日志工具类
class Loggers:
    """日志管理类，单例模式"""
    _logger = None

    @classmethod
    def get_logger(cls, log_dir: str = './logs') -> 'logging.Logger':
        if cls._logger is None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
            cls._logger = logging.getLogger('DPRW_Engine')
            cls._logger.setLevel(logging.INFO)
            cls._logger.handlers.clear()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(name)s] %(message)s")
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)
        return cls._logger

class ImageQualityMetrics:  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    FUNCTION = "compute_metrics"
    CATEGORY = "DPRW/Quality_Metrics"

    def to_pil(self,image):
        if isinstance(image, Image.Image):
            return image

        if hasattr(image, "cpu"):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        while image_np.ndim > 3:
            image_np = np.squeeze(image_np, axis=0)
        
        if image_np.ndim == 3:
            if image_np.shape[0] in [1, 3] and image_np.shape[-1] not in [1, 3]:
                image_np = np.transpose(image_np, (1, 2, 0))
        
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            else:
                image_np = image_np.clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(image_np)


    def compute_metrics(self, image1, image2):
        image1 = self.to_pil(image1)
        image2 = self.to_pil(image2)

        if image1.size != image2.size:
            common_width = min(image1.width, image2.width)
            common_height = min(image1.height, image2.height)
            image1 = image1.resize((common_width, common_height), resample=Image.BICUBIC)
            image2 = image2.resize((common_width, common_height), resample=Image.BICUBIC)
        
        np_img1 = np.array(image1).astype(np.float32)
        np_img2 = np.array(image2).astype(np.float32)
        
        mse = np.mean((np_img1 - np_img2) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * math.log10((255 ** 2) / mse)
        
        min_side = min(np_img1.shape[0], np_img1.shape[1])
        if min_side < 7:
            win_size = min_side if min_side % 2 == 1 else max(1, min_side - 1)
        else:
            win_size = 7
        
        if np_img1.ndim == 3 and np_img1.shape[-1] in [1, 3, 4]:
            ssim = structural_similarity(np_img1, np_img2, data_range=np_img2.max()-np_img2.min(),
                                         win_size=win_size, channel_axis=-1)
        else:
            ssim = structural_similarity(np_img1, np_img2, data_range=np_img2.max()-np_img2.min(),
                                         win_size=win_size)
        
        psnr_str = f"{psnr:.2f} dB"
        ssim_str = f"{ssim:.4f}"
        mse_str = f"{mse:.2f}"
        
        return (psnr_str, ssim_str, mse_str)
    
class DifferenceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 30, "min": 0.0, "max": 50.0, "step": 0.1}),
                "add_color": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_difference"
    CATEGORY = "DPRW/Difference"

    def generate_difference(self, image1, image2, scale_factor=1.0,add_color=False ):
        # Convert PIL images to NumPy arrays (OpenCV format)
        img1_np = np.array(image1.convert('RGB'))
        img2_np = np.array(image2.convert('RGB'))
        
        # OpenCV uses BGR format, so convert from RGB
        img1_cv = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
        
        # Make sure both images are the same size
        if img1_cv.shape != img2_cv.shape:
            # Resize the second image to match the first
            img2_cv = cv2.resize(img2_cv, (img1_cv.shape[1], img1_cv.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1_cv, img2_cv)
        
        # Apply scaling if requested
        if scale_factor != 1.0:
            diff = cv2.convertScaleAbs(diff, alpha=scale_factor, beta=0)
        
        # Apply colormap for better visualization 
        if add_color:
            diff_ = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        else:
        # Convert to grayscale
            diff_ = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Convert back to RGB for PIL
        diff_rgb = cv2.cvtColor(diff_, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(diff_rgb)

    def save_difference(self, diff, path):
        cv2.imwrite(path, diff)



class AttackSimulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "attack_type": (["rotation", "scaling", "resizedcrop", "erasing", 
                               "brightness", "contrast", "blurring", "noise", 
                               "compression", "horizontal_flip", "vertical_flip", 
                               "togray", "randomcrop", "invert"],),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_attack"
    CATEGORY = "DPRW/Attack"

    
    def to_pil(self, image):
        """将 ComfyUI 图像转换为 PIL 图像"""
        if isinstance(image, Image.Image):
            return image
            
        # 处理 PyTorch 张量
        if torch.is_tensor(image):
            if image.dim() == 4:  # [B, H, W, C]
                if image.size(0) == 1:
                    image = image.squeeze(0)  # [H, W, C]
                else:
                    raise ValueError("Batch size > 1 not supported")
                    
            # 确保图像是 [H, W, C] 格式
            if image.shape[-1] not in [1, 3, 4]:
                image = image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                
            # 转换为 NumPy
            image_np = image.cpu().numpy()
            
            # 归一化到 [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
                
            # 创建 PIL 图像
            if image_np.shape[-1] == 1:
                return Image.fromarray(image_np.squeeze(-1), mode='L')
            elif image_np.shape[-1] == 3:
                return Image.fromarray(image_np, mode='RGB')
            elif image_np.shape[-1] == 4:
                return Image.fromarray(image_np, mode='RGBA')
        
        # 处理 NumPy 数组
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
            
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def to_tensor(self, pil_image):
        """将 PIL 图像转换回 ComfyUI 可用的张量格式"""
        # 转换为 NumPy 数组
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 确保是 RGB
        if len(np_image.shape) == 2:  # 灰度图
            np_image = np.stack([np_image] * 3, axis=-1)
            
        # 转换为 PyTorch 张量，保持 [H, W, C] 格式
        tensor = torch.from_numpy(np_image).float()
        
        # 添加批次维度 [1, H, W, C]
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def relative_strength_to_absolute(self, strength, distortion_type):
        """将相对强度转换为绝对值"""
        # 定义各种失真方式的强度范围
        distortion_strength_paras = {
            "rotation": (0, 360),
            "scaling": (0, 1),
            "resizedcrop": (1, 0.1),
            "erasing": (0, 1),
            "brightness": (1, 4),
            "contrast": (1, 4),
            "blurring": (0, 4),
            "noise": (0, 0.5),
            "compression": (100, 0),
            "horizontal_flip": (0, 0),
            "vertical_flip": (0, 0),
            "togray": (0, 0),
            "randomcrop": (1, 0),
            "invert": (0, 0)
        }
        
        assert 0 <= strength <= 1
        min_val, max_val = distortion_strength_paras[distortion_type]
        abs_strength = strength * (max_val - min_val) + min_val
        
        # 确保在有效范围内
        abs_strength = max(abs_strength, min(min_val, max_val))
        abs_strength = min(abs_strength, max(min_val, max_val))
        
        return abs_strength
    
    def apply_single_attack(self, image, attack_type, strength):
        """应用单一类型的攻击"""
        abs_strength = self.relative_strength_to_absolute(strength, attack_type)
        
        if attack_type == "rotation":
            return F.rotate(image, abs_strength)
            
        elif attack_type == "scaling":
            new_size = (int(image.width * abs_strength), int(image.height * abs_strength))
            return image.resize(new_size, Image.Resampling.LANCZOS)
            
        elif attack_type == "resizedcrop":
            i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(abs_strength, abs_strength), ratio=(1, 1))
            return F.resized_crop(image, i, j, h, w, image.size)
            
        elif attack_type == "erasing":
            # 转换为张量进行操作
            tensor = T.ToTensor()(image).unsqueeze(0)
            i, j, h, w, v = T.RandomErasing.get_params(tensor, scale=(abs_strength, abs_strength), ratio=(1, 1), value=[0])
            tensor = F.erase(tensor, i, j, h, w, v)
            return T.ToPILImage()(tensor.squeeze(0))
            
        elif attack_type == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(abs_strength)
            
        elif attack_type == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(abs_strength)
            
        elif attack_type == "blurring":
            return image.filter(ImageFilter.GaussianBlur(int(abs_strength)))
            
        elif attack_type == "noise":
            # 转换为张量进行操作
            tensor = T.ToTensor()(image)
            noise = torch.randn_like(tensor) * abs_strength
            noisy_tensor = (tensor + noise).clamp(0, 1)
            return T.ToPILImage()(noisy_tensor)
            
        elif attack_type == "compression":
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=int(abs_strength))
            return Image.open(buffered)
            
        elif attack_type == "horizontal_flip":
            return F.hflip(image)
            
        elif attack_type == "vertical_flip":
            return F.vflip(image)
            
        elif attack_type == "togray":
            return image.convert("L").convert("RGB")
            
        elif attack_type == "randomcrop":
            i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(abs_strength, abs_strength), ratio=(1, 1))
            cropped = F.crop(image, i, j, h, w)
            # 创建一个与原图相同大小的全黑背景图像
            black_image = Image.new("RGB", image.size)
            # 将裁剪后的图像粘贴到黑背景的对应位置
            black_image.paste(cropped, (j, i))
            return black_image
            
        elif attack_type == "invert":
            return ImageOps.invert(image)
            
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def apply_attack(self, image, attack_type, strength, seed):
        """应用选定的攻击"""
        set_random_seed(seed)
        
        # 转换为 PIL 图像
        if not isinstance(image, Image.Image):
            image = self.to_pil(image)
        
        # 应用攻击
        attacked_image = self.apply_single_attack(image, attack_type, strength)
        
        # 转换回 ComfyUI 图像格式
        return self.to_tensor(attacked_image)




class DPRMetadataGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "prompt": ("STRING", {"default": "A cat sitting on a chair"}),
                "author_name": ("STRING", {"default": "lthero"}),
                "author_id": ("STRING", {"default": "02025HERO"}),
                "model_name": ("STRING", {"default": "flux"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json_string", "cbor_string")
    FUNCTION = "generate_metadata"
    CATEGORY = "DPRW/metadata"

    def generate_metadata(self, width, height, prompt, author_name,author_id, model_name, seed):
        # Get current UTC time and format as ISO 8601
        current_time = datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Create metadata dictionary
        metadata = {
            "author": {
                "name": author_name,
                "id": author_id
            },
            "parameters": {
                "model_name": model_name,
                "prompt": prompt,
                "width": width,
                "height": height,
                "timestamp": current_time,
                "seed": seed
            },
        }
        
        # Generate compressed JSON string (no whitespace)
        json_str = json.dumps(metadata, separators=(',', ':'))
        
        # Generate CBOR string
        cbor_bytes = cbor2.dumps(metadata)
        cbor_str = cbor_bytes.hex()  # Convert bytes to hex string for output
        
        return (json_str, cbor_str)


