from .DPRW_utils import *
class TreeRingWatermarker:
    """
    TreeRingWatermarker 类：封装生成树环风格水印噪声所需的函数。
    包括生成圆形掩码、生成水印图案、生成掩码、以及将水印注入 latent 中。
    """

    def __init__(self, device: str = "cuda",log_dir: str = './logs'):
        self.device = device
        self.logger = logging.getLogger(log_dir)

    def circle_mask(self, size: int = 64, r: int = 10, r_skip: int = 0, x_offset: int = 0, y_offset: int = 0):
        """
        生成圆形掩码。如果 r_skip 大于 0，则返回环形（外圆减内圆）的掩码。
        """
        x0 = size // 2 + x_offset
        y0 = size // 2 + y_offset
        y, x = np.ogrid[:size, :size]
        # 为了使 y 从上到下变化，对 y 轴进行反转
        y = y[::-1]
        mask_outer = (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2
        if r_skip > 0:
            mask_inner = (x - x0) ** 2 + (y - y0) ** 2 <= r_skip ** 2
            mask = mask_outer & ~mask_inner
        else:
            mask = mask_outer
        return mask

    def get_watermarking_pattern(self, w_pattern: str, w_radius: int):
        """
        根据 w_pattern 选择生成水印图案：
          - 若 w_pattern 中包含 "ring"，则对初始噪声进行环形注入
          - 若 w_pattern 为 "zeros"，则生成全 0 的图案
          - 其它情况则直接返回原始随机噪声
        """
        if 'ring' in w_pattern:
            gt_patch = self.init_latent.clone()
            gt_patch_tmp = gt_patch.clone()
            # 从 w_radius 到 1，每个半径的掩码进行注入
            for i in range(w_radius, 0, -1):
                mask_np = self.circle_mask(size=self.init_latent.shape[-1], r=i)
                mask_tensor = torch.tensor(mask_np, device=self.device)
                for j in range(self.init_latent.shape[1]):
                    # 按照固定位置注入一个常数，此处取第一个样本第 j 通道的一个值
                    gt_patch[:, j, mask_tensor] = gt_patch_tmp[0, j, 0, i].item()
        elif 'zeros' in w_pattern:
            gt_patch = self.init_latent * 0
        else:
            gt_patch = self.init_latent
        return gt_patch

    def get_watermarking_mask(self, w_radius: int, w_radius_skip: int, w_channel: int):
        """
        为给定的 init_latent 生成水印掩码。
        当 w_channel 为 -1 时，所有通道生效；否则只对指定通道生效。
        """
        watermarking_mask = torch.zeros_like(self.init_latent, dtype=torch.bool, device=self.device)
        np_mask = self.circle_mask(size=self.init_latent.shape[-1], r=w_radius, r_skip=w_radius_skip)
        mask_tensor = torch.tensor(np_mask, device=self.device)
        if w_channel == -1:
            watermarking_mask[:] = mask_tensor
        else:
            watermarking_mask[:, w_channel] = mask_tensor
        return watermarking_mask
    # def inject_watermark(self,watermarking_mask, gt_patch, w_injection, channel):
    #     init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(self.init_latent), dim=(-1, -2))
    #     print(self.init_latent.dtype)
    #     print(gt_patch.dtype)
    #     watermarking_mask = watermarking_mask.to(torch.float32)
    #     print(watermarking_mask.dtype)
    #     print(self.init_latent.device)
    #     print(gt_patch.device)
    #     print(watermarking_mask.device)
    #     if w_injection == 'complex':
    #         init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    #     elif w_injection == 'seed':
    #         init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
    #         return init_latents_w
    #     else:
    #         raise NotImplementedError(f'w_injection: {w_injection}')

    #     init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
    #     return init_latents_w

    def inject_watermark(self, watermarking_mask, gt_patch, w_injection, channel):
        # 先计算 FFT：进行中心化变换
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(self.init_latent), dim=(-1, -2)).to(self.device)
        self.init_latent = self.init_latent.to(self.device)
        gt_patch = gt_patch.to(self.device)
        # 打印相关信息，注意这里将 watermarking_mask 转换为 float32
        print("Init latent dtype:", self.init_latent.dtype)
        print("GT patch dtype:", gt_patch.dtype)
        watermarking_mask = watermarking_mask.to(torch.float32)
        print("Watermarking mask dtype (after conversion):", watermarking_mask.dtype)
        print("Init latent device:", self.init_latent.device)
        print("GT patch device:", gt_patch.device)
        print("Watermarking mask device:", watermarking_mask.device)
        
        if w_injection == 'complex':
            # 使用浮点型掩码进行加权混合注入：
            # 对于掩码为1的地方，将替换为对应 gt_patch 的值；为0的地方保持原值
            init_latents_w_fft = init_latents_w_fft * (1 - watermarking_mask) + gt_patch * watermarking_mask
        elif w_injection == 'seed':
            # 如果是 seed 注入，首先复制一份未处理的 latent
            init_latents_w = self.init_latent.clone()
            init_latents_w = init_latents_w * (1 - watermarking_mask) + gt_patch * watermarking_mask
            return init_latents_w
        else:
            raise NotImplementedError(f'w_injection: {w_injection}')
        
        # 将修改后的频域结果通过逆FFT转换回时域，并取实数部分
        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
        return init_latents_w

    # def inject_watermark(self, watermarking_mask: torch.Tensor, 
    #                      gt_patch: torch.Tensor, w_injection: str, channel: int):
    #     """
    #     将水印注入到 latent 中：
    #       - 若 w_injection 为 "complex"，则在频域进行注入
    #       - 若 w_injection 为 "seed"，则直接在时域替换
    #     """
    #     gt_patch=gt_patch.to(self.init_latent.dtype).to(self.device)
    #     watermarking_mask=watermarking_mask.to(self.init_latent.dtype).to(self.device)
    #     if w_injection == 'complex':
    #         latent_fft = torch.fft.fftshift(torch.fft.fft2(self.init_latent), dim=(-1, -2)).to(self.device)
    #         latent_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    #         out_latent = torch.fft.ifft2(torch.fft.ifftshift(latent_fft, dim=(-1, -2))).real.to(self.device)
    #     elif w_injection == 'seed':
    #         self.init_latent[watermarking_mask] = gt_patch[watermarking_mask].clone()
    #         out_latent = self.init_latent
    #     else:
    #         raise NotImplementedError(f'w_injection type "{w_injection}" is not implemented')
    #     return out_latent

    # 添加一个对noise进行处理的函数
    def _get_init_noise(self, noise: torch.Tensor, )->torch.Tensor:
        if self.latent_channels == 16 and noise.size(1) != 16:
            noise = torch.randn(1, self.latent_channels, noise.size(2), noise.size(3), device=self.device, dtype=noise.dtype)
            self.logger.warning(f"[TreeRingWatermarker] Embed watermark - Using NEW random noise: {noise.shape}")
        elif self.latent_channels == 4:
            # 先判断噪声是否是空值，如果是则重新生成
            samples = noise.cpu().numpy().flatten()
            _, p_value = kstest(samples, 'norm', args=(0, 1))
            if np.var(samples) == 0 or p_value < 0.05: 
                self.logger.warning(f"[TreeRingWatermarker] p_value {p_value}")
                self.logger.warning("[TreeRingWatermarker] Embed watermark - Noise variance is 0 or p-value<0.05 using new random noise")
                noise=torch.randn(1,noise.size(1),noise.size(2),noise.size(3),device=self.device,dtype=noise.dtype)
                self.logger.warning(f"[TreeRingWatermarker] Embed watermark - Using NEW random noise: {noise.shape}")
            else:
                self.logger.info(f"[TreeRingWatermarker] Embed watermark - The noise is not empty and can be used directly")
        return noise

    def create_tree_ring_latent(self, init_latent: torch.Tensor, seed: int, latent_channels_num: int,
                                w_channel: int, w_radius: int, w_radius_skip: int, w_pattern: str,
                                w_injection: str = 'complex'):
        """
        根据输入参数生成树环水印 latent：
          1. 设置随机种子
          2. 根据输入的宽高生成初始 latent，尺寸为 (1, latent_channels, height//8, width//8)
          3. 调用内部方法生成水印图案与掩码，并将水印注入 latent 中
        """
        # set_random_seed(seed)
        self.latent_channels=latent_channels_num
        self.init_latent = self._get_init_noise(init_latent)
        gt_patch = self.get_watermarking_pattern( w_pattern, w_radius  )
        watermarking_mask = self.get_watermarking_mask( w_radius, w_radius_skip, w_channel)
        watermarked_latent = self.inject_watermark( watermarking_mask, gt_patch, w_injection, w_channel)
        self.logger.info(f"[TreeRingWatermarker] Embed watermark - watermarked_latent: {watermarked_latent.shape}")
        return watermarked_latent,self.init_latent

class TreeRingLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "init_latent": ("LATENT",),
            "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            "message": ("STRING", {"default": "lthero"}),
            "message_length": ("INT", {"default": 32, "min": -1, "max": 81920}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "latent_channels_num": ("INT", {"default": 4, "min": 4, "max": 16}),
            "w_channel": ("INT", {"default": 0, "min": -1, "max": 3}),
            "w_radius": ("INT", {"default": 10, "min": 1, "max": 100}),
            "w_radius_skip": ("INT", {"default": 0, "min": 0, "max": 100}),
            "w_pattern": (["zeros", "ring"],),
        }}

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    FUNCTION = "create_treering_latents"
    CATEGORY = "DPRW/latent/TreeRingNoise"

    def create_treering_latents(self,init_latent,  message, batch_size, use_seed, seed, 
                                message_length, latent_channels_num, w_channel, w_radius, w_radius_skip,
                                w_pattern):
        """
        根据输入参数生成 batch_size 个带有树环风格水印的 latent 样本。
        当 message_length 小于 0 时，则自动采用 message 的 bit 长度（单位：bit）。
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_seed:
            set_random_seed(seed)
        if message_length < 0:
            message_length = len(message) * 8
        watermarker = TreeRingWatermarker(device=device)
        # latent_list = []
        # 固定参数：w_injection 固定为 'complex'，w_pattern_const 固定为 0.0
        w_injection = 'complex'
        print(init_latent)
        init_latent=init_latent["samples"]
        # for _ in range(batch_size):
        watermarked_latent,init_latent = watermarker.create_tree_ring_latent(init_latent,seed,  latent_channels_num,
                                                         w_channel, w_radius, w_radius_skip,
                                                         w_pattern,  w_injection)
        # latent_list.append(watermarked_latent)
        # 拼接成 batch 后返回
        # latent_tensor = torch.cat(latent_list, dim=0)
        Message_length_str = f"Message length: {message_length}"
        return ({"samples": watermarked_latent},{"samples": init_latent}, Message_length_str)
