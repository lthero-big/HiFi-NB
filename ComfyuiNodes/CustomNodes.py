from .DPRW_utils import *
from .DPRW_Watermark import *
from .TreeRing import *
from .GaussianShading import *

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.model_management
import comfy.utils

class WatermarkOptimizerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clean_latent": ("LATENT",), # This is z_T_clean (initial clean noise)
                "watermarked_latent": ("LATENT",), # This is z_T_watermarked (initial watermarked noise)
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "num_steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "num_iterations": ("INT", {"default": 10, "min": 1, "max": 100}), # Optimization iterations per diffusion step
                "lambda_semantic": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "lambda_watermark": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.00001, "max": 1.0, "step": 0.0001}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "cfg_scale_for_loss": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.5}), # CFG scale used during loss calculation
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("optimized_latent",)
    FUNCTION = "optimize"
    CATEGORY = "latent/watermark" # You can change the category

    def optimize(self, model, clean_latent, watermarked_latent, positive, negative,
                 num_steps, num_iterations, lambda_semantic, lambda_watermark,
                 learning_rate, sampler_name, scheduler, seed, cfg_scale_for_loss):

        device = model.load_device

        trajectory_storage_device = device 

        z_T_clean_samples = clean_latent["samples"].clone()
        
        clean_denoised_trajectory = [] # To store x0 predictions from the clean run
        
        k_sampler_clean = comfy.samplers.KSampler(model, steps=num_steps, device=device, 
                                                  sampler=sampler_name, scheduler=scheduler, 
                                                  denoise=1.0, model_options=model.model_options)
        sigmas_for_loop = k_sampler_clean.sigmas.clone() 

        # Callback to store denoised predictions (x0) from the clean run
        # ComfyUI progress bar can be resource-intensive if updated too frequently from background threads.
        # Using simple print statements for feedback.
        def clean_trajectory_callback(step, x0, x, total_steps):
            clean_denoised_trajectory.append(x0.clone().to(trajectory_storage_device))
            # Simple progress print for the clean trajectory generation
            if (step + 1) % (max(1, total_steps // 10)) == 0 or step == total_steps - 1:
                print(f"Clean trajectory generation: Step {step + 1}/{total_steps}")

        print("Generating clean trajectory...")
        # The `noise` parameter to KSampler.sample is the initial noisy latent z_T
        _ = k_sampler_clean.sample(
            noise=z_T_clean_samples, 
            positive=positive,
            negative=negative,
            cfg=cfg_scale_for_loss, # Use consistent CFG for x0 predictions
            latent_image=z_T_clean_samples, 
            start_step=0,
            last_step=num_steps, 
            force_full_denoise=False,
            denoise_mask=None,
            sigmas=sigmas_for_loop.to(device), # Pass the sigmas explicitly
            callback=clean_trajectory_callback,
            disable_pbar=True, # Disable KSampler's own pbar, we'll use prints
            seed=seed
        )
        
        if not clean_denoised_trajectory:
            raise RuntimeError("Clean trajectory generation failed or produced no steps. Check num_steps.")
        
        print(f"Clean trajectory generated with {len(clean_denoised_trajectory)} denoised states.")

        # --- Process conditionings for the optimization loop's sampling_function call ---
        print("Processing conditionings for optimization loop...")
        processed_positive = comfy.sampler_helpers.convert_cond(positive)
        processed_negative = comfy.sampler_helpers.convert_cond(negative)

        print(f"processed_positive {processed_negative[0]['pooled_output'].shape}")

        # --- 2. Initialize optimized latent and optimizer ---
        z_T_watermarked_samples = watermarked_latent["samples"].clone()
        # Optimize in float32 for stability, and ensure it's on the correct device
        z_t_opt = z_T_watermarked_samples.clone().to(device=device, dtype=model.model_dtype()).requires_grad_(True)
        
        optimizer = torch.optim.AdamW([z_t_opt], lr=learning_rate, weight_decay=1e-6) # As in user's code

        # Prepare target for watermark loss (derived from the initial watermarked latent)
        with torch.no_grad():
            # Assuming watermark is embedded by making some latent values positive/negative.
            # Target for BCEWithLogitsLoss should be 0s and 1s.
            target_wm_binary_flat = (z_T_watermarked_samples.to(device=device, dtype=model.model_dtype()).flatten() > 0.0).float()

        print("Starting optimization of watermarked latent...")
        
        # --- 3. Main optimization loop (over diffusion timesteps/sigmas) ---
        # Effective number of steps for the loop, ensuring it doesn't exceed trajectory length
        # len(clean_denoised_trajectory) should be equal to num_steps if generation was successful
        loop_num_steps = min(num_steps, len(clean_denoised_trajectory)) 
        if loop_num_steps < num_steps:
             print(f"Warning: Effective optimization steps reduced to {loop_num_steps} due to clean trajectory length ({len(clean_denoised_trajectory)}).")

        if getattr(model.model, 'current_patcher', None) is None:
            print("Warning: model.model.current_patcher is None. Attempting to set it.")
            # This assumes 'model' is the ModelPatcher instance given to your node
            model.model.current_patcher = model

        for step_idx in range(loop_num_steps):
            # Current sigma for this diffusion step
            current_sigma = sigmas_for_loop[step_idx].unsqueeze(0).to(device)
            
            # Target denoised output from the clean trajectory for this step
            target_x0_clean_step = clean_denoised_trajectory[step_idx].to(device=device, dtype=torch.float32)

            # Inner optimization loop for the current timestep/sigma
            for iter_idx in range(num_iterations):
                optimizer.zero_grad()
                
                # Ensure z_t_opt (the variable being optimized) is in the correct dtype for the model
                input_latent_for_model = z_t_opt.to(model.model_dtype())

                # Get x0 prediction for the current z_t_opt at current_sigma using CFG
                # The `model` (ModelPatcher) handles its internal precision context.
                denoised_pred_opt = comfy.samplers.sampling_function(
                    model.model, # This is the ModelPatcher from ComfyUI
                    input_latent_for_model,
                    current_sigma,
                    processed_negative, # Unconditional conditioning
                    processed_positive, # Conditional conditioning
                    cfg_scale_for_loss, 
                    model.model_options # Pass the ModelPatcher's model_options
                ).to(dtype=torch.float32) # Ensure result is float32 for loss calculation
                
                # Semantic Loss: MSE between denoised prediction of optimized latent and denoised clean latent
                loss_semantic = F.mse_loss(denoised_pred_opt, target_x0_clean_step)

                # Watermark Loss: BCEWithLogits between current z_t_opt (flattened) and target binary watermark
                # The values in z_t_opt are treated as logits for the binary watermark decision.
                loss_watermark = F.binary_cross_entropy_with_logits(
                    z_t_opt.flatten(), target_wm_binary_flat, reduction='mean'
                )
                
                total_loss = lambda_semantic * loss_semantic + lambda_watermark * loss_watermark

                if torch.isnan(total_loss):
                    print(f"Warning: NaN loss at opt_step {step_idx+1}, iter {iter_idx+1}. Sem: {loss_semantic.item():.4f}, WM: {loss_watermark.item():.4f}. Skipping update.")
                    break # Break inner loop for this sigma if NaN occurs
                
                total_loss.backward()
                # Gradient clipping (as in user's reference code)
                torch.nn.utils.clip_grad_norm_([z_t_opt], max_norm=1.0)
                optimizer.step()
            
            if (step_idx + 1) % (max(1, loop_num_steps // 10)) == 0 or step_idx == loop_num_steps - 1:
                print(f"Optimization Step {step_idx + 1}/{loop_num_steps} completed. Last Total Loss: {total_loss.item():.4f} (Sem: {loss_semantic.item():.4f}, WM: {loss_watermark.item():.4f})")
            
        # Return the optimized latent, cast back to the original latent's dtype
        optimized_latent_dict = {"samples": z_t_opt.detach().to(z_T_watermarked_samples.dtype)}
        
        print("Optimization finished.")
        return (optimized_latent_dict,)




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
                "resampling_method": ("STRING", {"default": "quantile", "choices": ["inverse", "quantile", "resampling"]}),
            }
        }
    # 返回原二进制、提取的二进制、解码消息、正确率、latent
    RETURN_TYPES = ("STRING", "STRING","STRING","STRING","LATENT")  
    FUNCTION = "extract"
    CATEGORY = "DPRW/extractor"

    def extract(self, latents, key, nonce,message, message_length, window_size, watermarkMethod, resampling_method):
        """从潜在表示中提取水印"""
        if not isinstance(latents, dict) or "samples" not in latents:
            raise ValueError("latents must be a dictionary containing 'samples' key")
        # print(f"message_length {message_length}",type(message_length))

        noise = latents["samples"]

        # save_latent_data(noise, "ReversedNoiseFromDPRWExtractor")
        if message != "":
            message_length = len(message) * 8
        elif message_length != "":
            message_length = int(message_length)    
        else:
            raise ValueError("message_length or message must be provided, if you only know the message length, please provide the length")
        
        if watermarkMethod == "DPRW":
            dprw = DPRWatermark(key, nonce,latent_channels=noise.shape[1])
            print(f"resampling_method:!!!!!!!!!!! {resampling_method}")
            extracted_msg_bin, extracted_msg_str, extracted_msg_bin_segments = dprw.extract_watermark_original(noise, message_length, window_size, resampling_method)
            if message != "":
                orig_bin,accuracy = dprw.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str, extracted_msg_bin_segments)
            else:
                orig_bin,accuracy = extracted_msg_bin,-1
        elif watermarkMethod == "GS":
            gs = GSWatermark(key, nonce,latent_channels=noise.shape[1])
            extracted_msg_bin, extracted_msg_str, extracted_msg_bin_segments = gs.extract_watermark(noise, message_length, window_size)
            if message != "":
                orig_bin,accuracy = gs.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
            else:
                orig_bin,accuracy = extracted_msg_bin,-1
        else:
            raise ValueError(f"Invalid watermarkMethod: {watermarkMethod} ,must be DPRW or GS")
        Extracted_bin_str = extracted_msg_bin
        Extracted_msg_str = extracted_msg_str
        if accuracy != -1:
            Accuray_str = f"{accuracy*100:.2f}"
            Orig_bin_str = f"{orig_bin}"
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





# class DPRWatermarkOptimizer:
#     """ComfyUI版本的水印优化器节点"""
    
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model": ("MODEL",),
#                 "clean_latent": ("LATENT",),  # Z_T 干净噪声
#                 "watermarked_latent": ("LATENT",),  # Z_T^w 水印噪声
#                 "positive": ("CONDITIONING",),
#                 "negative": ("CONDITIONING",),
#                 "num_steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
#                 "num_iterations": ("INT", {"default": 10, "min": 1, "max": 100}),
#                 "lambda_semantic": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
#                 "lambda_watermark": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
#                 "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
#                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
#                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
#                 # "sampler": ("SAMPLER",),
#                 # "sigmas": ("SIGMAS",),
#                 "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
#             },
#             "optional": {
#                 "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
#                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#             }
#         }

#     RETURN_TYPES = ("LATENT", "STRING")
#     RETURN_NAMES = ("optimized_latent", "optimization_log")
#     FUNCTION = "optimize_watermark"
#     CATEGORY = "DPRW/optimizer"

#     def __init__(self):
#         self.logger = Loggers.get_logger()

#     def get_scheduler_and_sampler(self, model, sampler_name, scheduler, num_steps):
#         """获取调度器和采样器"""
#         device = comfy.model_management.get_torch_device()
        
#         # 获取调度器
#         scheduler_obj = comfy.samplers.calculate_sigmas(
#             model.get_model_object("model_sampling"), scheduler, num_steps
#         )
        
#         # 获取采样器
#         sampler = comfy.samplers.KSampler(
#             model, steps=num_steps, device=device, sampler=sampler_name, 
#             scheduler=scheduler, denoise=1.0, model_options=model.model_options
#         )
        
#         return scheduler_obj, sampler

#     def encode_conditioning(self, model, positive, negative):
#         """编码条件信息"""
#         # ComfyUI中的conditioning已经是编码好的
#         return positive, negative

#     def generate_clean_trajectory(self, model, clean_latent, positive, negative, 
#                                 sampler, scheduler_obj, num_steps, cfg):
#         """生成干净轨迹"""
#         device = comfy.model_management.get_torch_device()
#         clean_trajectory = {}
        
#         latents = clean_latent["samples"].clone().to(device)
#         batch_size = latents.shape[0]
        
#         with torch.no_grad():
#             # 设置调度器时间步
#             sigmas = scheduler_obj
            
#             for i, sigma in enumerate(sigmas[:-1]):  # 排除最后一个sigma
#                 clean_trajectory[i] = latents.clone()
                
#                 # 预测噪声 - 使用ComfyUI的模型预测接口
#                 sigma_tensor = torch.full((batch_size,), sigma, device=device, dtype=latents.dtype)
                
#                 # 获取模型预测
#                 model_output = model.model.apply_model(
#                     latents, sigma_tensor, cond=positive, uncond=negative, 
#                     cfg=cfg, model_options=model.model_options
#                 )
                
#                 # 更新latents - 使用调度器的步进方法
#                 if i < len(sigmas) - 2:
#                     next_sigma = sigmas[i + 1]
#                     latents = self.scheduler_step(latents, model_output, sigma, next_sigma)
                
#             clean_trajectory[len(sigmas) - 1] = latents.clone()  # 最终结果
            
#         return clean_trajectory

#     def scheduler_step(self, x, model_output, sigma, next_sigma):
#         """调度器步进 - 简化的DDIM步进"""
#         alpha_prod_t = 1 / (sigma**2 + 1)
#         alpha_prod_t_prev = 1 / (next_sigma**2 + 1) if next_sigma > 0 else 1.0
        
#         beta_prod_t = 1 - alpha_prod_t
        
#         # DDIM步进公式
#         pred_original_sample = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        
#         alpha_prod_t_prev = torch.clamp(alpha_prod_t_prev, min=1e-8)
        
#         pred_sample = (
#             alpha_prod_t_prev**0.5 * pred_original_sample +
#             (1 - alpha_prod_t_prev)**0.5 * model_output
#         )
        
#         return pred_sample

#     def optimize_watermark(self, model, clean_latent, watermarked_latent, positive, negative,
#                          num_steps, num_iterations, lambda_semantic, lambda_watermark, 
#                          learning_rate, sampler_name, scheduler, cfg, seed=42, denoise=1.0):
#         """主要的水印优化函数"""
        
#         device = comfy.model_management.get_torch_device()
        
#         # 移动数据到设备
#         z_T_clean = clean_latent["samples"].clone().to(device)
#         z_T_watermarked = watermarked_latent["samples"].clone().to(device)
        
#         self.logger.info(f"开始水印优化过程...")
#         self.logger.info(f"Clean latent shape: {z_T_clean.shape}")
#         self.logger.info(f"Watermarked latent shape: {z_T_watermarked.shape}")
        
#         # 获取调度器和采样器
#         scheduler_obj, sampler = self.get_scheduler_and_sampler(model, sampler_name, scheduler, num_steps)
        
#         # 编码条件信息
#         pos_cond, neg_cond = self.encode_conditioning(model, positive, negative)
        
#         # 生成干净轨迹
#         self.logger.info("生成参考轨迹（干净）...")
#         clean_trajectory = self.generate_clean_trajectory(
#             model, clean_latent, pos_cond, neg_cond, sampler, scheduler_obj, num_steps, cfg
#         )
        
#         # 开始优化过程
#         self.logger.info("开始优化水印轨迹...")
        
#         # 初始化优化变量
#         z_t_opt = z_T_watermarked.clone().float().detach().requires_grad_(True)
#         optimizer = torch.optim.AdamW([z_t_opt], lr=learning_rate, weight_decay=1e-6)
        
#         sigmas = scheduler_obj
#         loss_history = {'semantic': [], 'watermark': [], 'total': []}
        
#         optimization_log = []
#         optimization_log.append(f"开始优化，总时间步数: {len(sigmas) - 1}")
        
#         for step_idx, sigma in enumerate(sigmas[:-1]):
#             step_log = f"优化时间步 {step_idx+1}/{len(sigmas)-1}, sigma={sigma:.6f}"
#             self.logger.info(step_log)
#             optimization_log.append(step_log)
            
#             for iter_idx in range(num_iterations):
#                 optimizer.zero_grad()
                
#                 # 检查NaN
#                 if torch.isnan(z_t_opt).any():
#                     self.logger.error("z_t_opt包含NaN，重新初始化")
#                     z_t_opt = z_T_watermarked.clone().float().detach().requires_grad_(True)
#                     optimizer = torch.optim.AdamW([z_t_opt], lr=learning_rate, weight_decay=1e-6)
#                     break
                
#                 # 确保索引不越界
#                 if step_idx >= len(clean_trajectory) - 1:
#                     break
                
#                 # 转换数据类型进行模型计算
#                 z_t_opt_model = z_t_opt.to(device)
#                 batch_size = z_t_opt_model.shape[0]
#                 sigma_tensor = torch.full((batch_size,), sigma, device=device, dtype=z_t_opt_model.dtype)
                
#                 # 预测噪声
#                 model_output = model.model.apply_model(
#                     z_t_opt_model, sigma_tensor, cond=pos_cond, uncond=neg_cond,
#                     cfg=cfg, model_options=model.model_options
#                 )
                
#                 # 预测下一步
#                 if step_idx < len(sigmas) - 2:
#                     next_sigma = sigmas[step_idx + 1]
#                     z_t_minus_1_pred = self.scheduler_step(z_t_opt_model, model_output, sigma, next_sigma)
#                 else:
#                     z_t_minus_1_pred = z_t_opt_model
                
#                 # 计算语义损失
#                 if step_idx + 1 < len(clean_trajectory):
#                     target_t_minus_1 = clean_trajectory[step_idx + 1].float().detach()
#                 else:
#                     target_t_minus_1 = clean_trajectory[len(clean_trajectory) - 1].float().detach()
                
#                 loss_semantic = F.mse_loss(z_t_minus_1_pred.float(), target_t_minus_1)
                
#                 # 计算水印损失
#                 with torch.no_grad():
#                     z_T_watermarked_float = z_T_watermarked.float().detach()
#                     target_wm_binary = (z_T_watermarked_float > 0).float()
                
#                 loss_watermark = F.binary_cross_entropy_with_logits(
#                     z_t_opt, target_wm_binary, reduction='mean'
#                 )
                
#                 # 总损失
#                 total_loss = lambda_semantic * loss_semantic + lambda_watermark * loss_watermark
                
#                 # 检查损失是否为NaN
#                 if torch.isnan(total_loss):
#                     self.logger.warning(f"检测到NaN损失，跳过此次优化")
#                     break
                
#                 total_loss.backward()
                
#                 # 梯度裁剪
#                 torch.nn.utils.clip_grad_norm_([z_t_opt], max_norm=1.0)
                
#                 optimizer.step()
                
#                 # 记录损失历史
#                 if step_idx < 5:  # 只记录前几步的损失历史
#                     loss_history['semantic'].append(loss_semantic.item())
#                     loss_history['watermark'].append(loss_watermark.item())
#                     loss_history['total'].append(total_loss.item())
                
#                 if iter_idx % 5 == 0:
#                     iter_log = (f"  迭代 {iter_idx+1}/{num_iterations}, "
#                               f"语义损失: {loss_semantic.item():.6f}, "
#                               f"水印损失: {loss_watermark.item():.6f}, "
#                               f"总损失: {total_loss.item():.6f}")
#                     self.logger.info(iter_log)
#                     if step_idx < 3:  # 只记录前几步的详细日志
#                         optimization_log.append(iter_log)
                
#                 # 重新初始化优化变量
#                 if iter_idx < num_iterations - 1:
#                     with torch.no_grad():
#                         z_t_opt_new = z_t_opt.clone().detach().requires_grad_(True)
#                     z_t_opt = z_t_opt_new
#                     optimizer = torch.optim.AdamW([z_t_opt], lr=learning_rate, weight_decay=1e-6)
        
#         # 返回优化后的潜在表示
#         final_latents = z_t_opt.detach().to(device)
        
#         # 构建输出latent字典
#         optimized_latent = {
#             "samples": final_latents
#         }
        
#         # 复制其他可能的元数据
#         for key in watermarked_latent:
#             if key != "samples":
#                 optimized_latent[key] = watermarked_latent[key]
        
#         optimization_log.append("优化完成!")
#         log_string = "\n".join(optimization_log)
        
#         self.logger.info("水印优化完成!")
        
#         return (optimized_latent, log_string)