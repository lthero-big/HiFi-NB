from .DPRW_utils import *
from .DPRW_Watermark import *
from .TreeRing import *
from .GaussianShading import *


class DPRExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
                "message": ("STRING", {"default": "lthero"}),
                "message_length": ("STRING", {"default": ""}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "watermarkMethod": ("STRING", {"default": "DPRW","options":["GS","DPRW"]}),
            }
        }
    # 返回原二进制、提取的二进制、解码消息、正确率、latent
    RETURN_TYPES = ("STRING", "STRING","STRING","STRING","LATENT")  
    FUNCTION = "extract"
    CATEGORY = "DPRW/extractor"

    def extract(self, latents, key, nonce,message, message_length, window_size,watermarkMethod):
        """从潜在表示中提取水印"""
        if not isinstance(latents, dict) or "samples" not in latents:
            raise ValueError("latents must be a dictionary containing 'samples' key")
        # print(f"message_length {message_length}",type(message_length))
        noise = latents["samples"]
        if message != "":
            message_length = len(message) * 8
        elif message_length != "":
            message_length = int(message_length)    
        else:
            raise ValueError("message_length or message must be provided, if you only know the message length, please provide the length")
        
        if watermarkMethod == "DPRW":
            dprw = DPRWatermark(key, nonce,latent_channels=noise.shape[1])
            extracted_msg_bin, extracted_msg_str = dprw.extract_watermark(noise, message_length, window_size)
            if message != "":
                orig_bin,accuracy = dprw.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
            else:
                orig_bin,accuracy = extracted_msg_bin,-1
        elif watermarkMethod == "GS":
            gs = GSWatermark(key, nonce,latent_channels=noise.shape[1])
            extracted_msg_bin, extracted_msg_str = gs.extract_watermark(noise, message_length, window_size)
            if message != "":
                orig_bin,accuracy = gs.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
            else:
                orig_bin,accuracy = extracted_msg_bin,-1
        else:
            raise ValueError(f"Invalid watermarkMethod: {watermarkMethod} ,must be DPRW or GS")
        Extracted_bin_str = f"{extracted_msg_bin}"
        Extracted_msg_str = f"{extracted_msg_str}"
        if accuracy != -1:
            Accuray_str = f"{accuracy:.2f} ({accuracy*100:.2f}%)"
            Orig_bin_str = f"Original binary: {orig_bin}"
        else:
            Orig_bin_str = f"Provide the original message"
            Accuray_str = f"Provide the original message to evaluate the accuracy"
        return (Orig_bin_str, Extracted_bin_str, Extracted_msg_str,Accuray_str,latents)

class DPRKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_dprw_noise": (["enable", "disable"],),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "watermarked_latent_noise": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }
    #  the first latent is denoised one, the second one is init noise latent
    RETURN_TYPES = ("LATENT","LATENT")
    FUNCTION = "sample"
    CATEGORY = "DPRW/sampling"
    
    def sample(self, model, use_dprw_noise, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, watermarked_latent_noise, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        """高级采样器，支持 DPRW 水印噪声"""
        force_full_denoise = return_with_leftover_noise != "enable"
        use_dprw = use_dprw_noise == "enable"
        disable_noise = add_noise == "disable"
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                               force_full_denoise=force_full_denoise, use_dprw=use_dprw, watermarked_latent_noise=watermarked_latent_noise)
