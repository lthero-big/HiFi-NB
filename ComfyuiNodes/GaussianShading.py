from .DPRW_utils import *

class GSWatermark:
    """
    GSWatermark 类：基于 Gaussian Shading 算法（改编自 Gaussian_Shading_chacha）生成水印噪声
    支持任意尺寸图片和最多16通道，采用 Gaussian_Shading_chacha 的投票机制和 threshold
    """
    def __init__(self, key_hex: str, nonce_hex: str, device: str = "cuda", use_seed: bool = True, seed: int = 42, 
                 watermark_length_bits: int = 64, latent_channels: int = 4, log_dir: str = './logs',
                 ch_factor: int = 1, hw_factor: int = 1, fpr: float = 0.01, user_number: int = 1000):
        # 验证并截取 key 和 nonce（nonce 固定为 12 字节）
        self.key = validate_hex(key_hex, 64, os.urandom(32))
        self.nonce = validate_hex(nonce_hex, 32, os.urandom(16)) # os.urandom(16) -- useful # bytes.fromhex(nonce_hex) # validate_hex(nonce_hex, 12, os.urandom(12))[:12]  # 截取前 12 字节
        self.device = device
        self.use_seed = use_seed
        self.seed = seed
        self.watermark_length = watermark_length_bits
        self.channels = min(latent_channels, 16)  # 最多支持 16 通道
        self.logger = Loggers.get_logger(log_dir)
        self.ch_factor = ch_factor
        self.hw_factor = hw_factor
        self.fpr = fpr
        self.user_number = user_number
        self.latent_width = 64
        self.latent_height = 64
        self.marklength = None  # 将在 init_noise 中计算
        self.threshold = None   # 将在 init_noise 中设置
        self.tau_onebit = None
        self.tau_bits = None
        try:
            self.watermark = torch.load('gs_watermark.pt', map_location=self.device)
        except FileNotFoundError:
            self.watermark = torch.randint(0, 2, (1, self.channels, self.latent_width, self.latent_height), device=self.device, dtype=torch.uint8)
            torch.save(self.watermark, 'gs_watermark.pt')

        self.logger.info(f"====================GS Watermark Initialized Begin====================")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")
        self.logger.info(f"Initialized - Channels: {self.channels}")
        self.logger.info(f"Initialized - ch_factor: {self.ch_factor}, hw_factor: {self.hw_factor}")
        self.logger.info(f"====================GS Watermark Initialized End====================")

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> torch.Tensor:
        """生成随机水印张量（基于 Gaussian_Shading_chacha）"""
        length_bits = message_length if message_length > 0 else min(self.watermark_length, total_blocks)
        latent_ch = self.channels // self.ch_factor
        self.marklength = latent_ch * self.latent_width * self.latent_height
        self.threshold = 1 if self.ch_factor == 1 and self.hw_factor == 1 else self.ch_factor * self.hw_factor * self.hw_factor // 2
        
        # 计算 tau_onebit 和 tau_bits
        for i in range(self.marklength):
            fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
            fpr_bits = betainc(i + 1, self.marklength - i, 0.5) * self.user_number
            if fpr_onebit <= self.fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= self.fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
        
        # 生成随机水印张量
        # length_bytes = length_bits // 8
        # msg_bytes = message.encode('utf-8')
        # padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        # repeats = total_blocks // length_bits
        # self.logger.info(f"Create watermark - total_blocks: {total_blocks}")
        # self.logger.info(f"Create watermark - repeats: {repeats}")
        
        
        
        # self.watermark_bin = padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)
        # self.watermark = torch.tensor(np.unpackbits(np.frombuffer(self.watermark_bin, dtype=np.uint8)), 
        #                               device=self.device, dtype=torch.uint8).reshape(1, latent_ch, self.latent_width, self.latent_height)
        # print watermark to hex


        self.logger.info(f"Create watermark - Message: {message}")
        self.logger.info(f"Create watermark - Message Length: {message_length}")
        self.logger.info(f"Create watermark - Watermark shape: {self.watermark.shape}")
        self.logger.info(f"Create watermark - Threshold: {self.threshold}, tau_onebit: {self.tau_onebit}, tau_bits: {self.tau_bits}")
        return self.watermark

    def _encrypt(self, watermark: torch.Tensor) -> np.ndarray:
        """加密水印（基于 Gaussian_Shading_chacha 的 ChaCha20，nonce 12 字节）"""
        # 将水印展平为二进制数组
        watermark_flat = watermark.flatten().cpu().numpy().astype(np.uint8)
        watermark_bytes = np.packbits(watermark_flat).tobytes()
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark_bytes) + encryptor.finalize()
        # 转换为二进制位
        m_bit = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))
        self.logger.info(f"Encrypt watermark - Encrypted bits length: {len(m_bit)}")
        return m_bit

    def _embed_watermark(self, encrypted_watermark_bits: np.ndarray, latent_height: int, latent_width: int) -> torch.Tensor:
        """将加密水印嵌入到潜在空间（基于 Gaussian_Shading_chacha 的截断正态分布）"""
        Z_s_T_array = torch.zeros((self.channels, latent_height, latent_width), dtype=torch.float32, device=self.device)
        total_length = self.channels * latent_height * latent_width
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]

        if self.use_seed:
            rng = np.random.RandomState(seed=self.seed)
        index = 0
        for i in range(min(len(encrypted_watermark_bits), total_length)):
            dec_mes = int(encrypted_watermark_bits[i])
            if not self.use_seed:
                z_s_T = torch.tensor(truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1]), dtype=torch.float32, device=self.device)
            else:
                z_s_T = torch.tensor(truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1], random_state=rng), 
                                   dtype=torch.float32, device=self.device)
            channel_index = index // (latent_height * latent_width)
            h_index = (index // latent_width) % latent_height
            w_index = index % latent_width
            Z_s_T_array[channel_index, h_index, w_index] = z_s_T
            index += 1

        self.logger.info(f"Embed watermark - Output shape: {Z_s_T_array.shape}")
        return Z_s_T_array

    def init_noise(self, message: str, Image_width: int, Image_height: int, window_size: int = 1) -> torch.Tensor:
        """初始化水印噪声（接口与 GSWatermark 一致）"""
        self.message = message
        self.latent_width = Image_width // self.hw_factor
        self.latent_height = Image_height // self.hw_factor
        total_blocks = self.channels * self.latent_width * self.latent_height
        message_length = len(message) * 8

        # 生成水印张量
        watermark = self._create_watermark(total_blocks, message, message_length)
        # 扩展水印到完整潜在空间
        watermark_expanded = watermark.repeat(1, self.ch_factor, self.hw_factor, self.hw_factor)
        # 加密水印
        encrypted_watermark_bits = self._encrypt(watermark_expanded)
        # 嵌入水印到潜在空间
        return self._embed_watermark(encrypted_watermark_bits, self.latent_height, self.latent_width)

    def extract_watermark(self, reversed_latents: torch.Tensor, message_length: int, window_size: int = 1) -> tuple:
        """提取水印（使用 Gaussian_Shading_chacha 的投票机制）"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()

        # 从潜在空间重建二进制位
        reversed_m = (reversed_latents > 0).int()
        reversed_m_flat = reversed_m.flatten().cpu().numpy().astype(np.uint8)
        m_reconstructed_bytes = np.packbits(reversed_m_flat).tobytes()
        sd_reconstructed = decryptor.update(m_reconstructed_bytes) + decryptor.finalize()
        sd_bit = np.unpackbits(np.frombuffer(sd_reconstructed, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, self.channels, 
                                                    reversed_latents.shape[2], 
                                                    reversed_latents.shape[3]).to(self.device, dtype=torch.uint8)

        # 使用 Gaussian_Shading_chacha 的投票机制
        ch_stride = self.channels // self.ch_factor
        hw_stride = reversed_latents.shape[2] // self.hw_factor
        ch_list = [ch_stride] * self.ch_factor
        hw_list = [hw_stride] * self.hw_factor
        split_dim1 = torch.cat(torch.split(sd_tensor, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        if self.threshold is None:
            self.threshold = self.ch_factor * self.hw_factor * self.hw_factor // 2
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1

        self.logger.info(f"Extract watermark - Vote shape: {vote.shape}")
        self.logger.info(f"Extract watermark - BA???: {(vote == self.watermark).float().mean().item()}")
        # 转换为二进制字符串
        reconstructed_bits = ''.join(str(int(b)) for b in vote.flatten().cpu().numpy())
        # 尝试解码为消息（与 GSWatermark 兼容）
        try:
            msg_bytes = bytes(int(reconstructed_bits[i:i + 8], 2) for i in range(0, min(message_length, len(reconstructed_bits)), 8))
            msg = msg_bytes.decode('utf-8', errors='replace')
        except:
            msg = ""
        segments = [reconstructed_bits[i:i + message_length] for i in range(0, len(reconstructed_bits), message_length) 
                    if len(reconstructed_bits[i:i + message_length]) == message_length]

        self.logger.info(f"Extract watermark - Reconstructed bits length: {len(reconstructed_bits)}")
        self.logger.info(f"Extract watermark - Extracted message: {msg}")
        return reconstructed_bits, msg, segments

    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str: str = "") -> tuple:
        """计算位准确率（与 GSWatermark 接口一致）"""
        self.logger.info(f"====================GS Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len if min_len > 0 else 0.0
        self.logger.info(f"Evaluation - self.watermark : {self.watermark.shape}")


        self.logger.info(f"Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"Evaluation - Extracted binary: {extracted_bin}")
        self.logger.info(f"Evaluation - Extracted binary length: {len(extracted_bin)}")
        self.logger.info(f"Evaluation - Bit accuracy: {accuracy}")
        self.logger.info(f"====================GS Watermark Evaluate End====================")
        return orig_bin, accuracy

class GSLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "channels": ("INT", {"default": 4, "min": 4, "max": 16, "step": 12}),
                "Image_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "Image_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f"}),
                "message": ("STRING", {"default": "lthero"}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "ch_factor": ("INT", {"default": 1, "min": 1, "max": 4}),
                "hw_factor": ("INT", {"default": 1, "min": 1, "max": 8}),
                "fpr": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "user_number": ("INT", {"default": 1000, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    FUNCTION = "create_gs_latents"
    CATEGORY = "DPRW/latent"

    def create_gs_latents(self, key, nonce, message, batch_size, use_seed, seed, Image_width, Image_height, 
                         channels, window_size, ch_factor, hw_factor, fpr, user_number):
        device = "cuda"
        message_length = len(message) * 8
        gs_watermark = GSWatermark(
            key_hex=key, nonce_hex=nonce, device=device, use_seed=bool(use_seed), seed=seed,
            watermark_length_bits=message_length, latent_channels=channels, 
            ch_factor=ch_factor, hw_factor=hw_factor, fpr=fpr, user_number=user_number
        )
        latent_list = [gs_watermark.init_noise(message, Image_width, Image_height, window_size) 
                      for _ in range(batch_size)]
        latent = torch.stack(latent_list)
        Message_length_str = f"Message length: {message_length}"
        return ({"samples": latent}, Message_length_str)