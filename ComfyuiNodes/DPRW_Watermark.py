from .DPRW_utils import *

class DPRWatermark:
    """DPRW 水印算法的核心实现"""
    def __init__(self, key_hex: str, nonce_hex: str,latent_channels:int=4,  device: str = "cuda",log_dir: str = './logs'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_channels = latent_channels
        self.key = validate_hex(key_hex, 64, os.urandom(32))
        self.nonce = validate_hex(nonce_hex, 32, os.urandom(16))
        self.logger = Loggers.get_logger(log_dir)
        self.logger.info(f"====================DPRW Watermark Begin====================")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        """生成水印字节"""
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        self.logger.info(f"[DPRWatermark]Create watermark - Message: {message}")
        self.logger.info(f"[DPRWatermark]Create watermark - Message Length: {message_length}")
        self.logger.info(f"[DPRWatermark]Create watermark - Watermark repeats: {repeats} times")
        return padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)

    def _encrypt(self, watermark: bytes) -> str:
        """加密水印并转换为二进制字符串"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _binarize_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """将噪声二值化"""
        return (torch.sigmoid(noise.to(self.device)) > 0.5).to(torch.uint8).flatten()

    def _embed_bits(self, binary: torch.Tensor, bits: str, window_size: int) -> torch.Tensor:
        """在 GPU 上嵌入水印位"""
        num_windows = len(binary) // window_size
        bits_tensor = torch.tensor([int(b) for b in bits[:num_windows]], dtype=torch.uint8, device=self.device)
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2
        flip_mask = window_sums != bits_tensor
        mid_idx = window_size // 2
        flip_indices = (torch.arange(num_windows, device=self.device) * window_size + mid_idx)[flip_mask]
        binary[flip_indices] = 1 - binary[flip_indices]
        return binary

    def _restore_noise(self, binary: torch.Tensor, shape: tuple, seed: int,original_noise:torch.Tensor) -> torch.Tensor:
        """还原高斯噪声"""
        # set_random_seed(seed)
        # noise = torch.randn(shape, device=self.device)
        binary_reshaped = binary.view(shape[1:]).to(self.device)
        original_binary = (torch.sigmoid(original_noise) > 0.5).to(torch.uint8).to(self.device)
        mask = binary_reshaped != original_binary
        u = torch.rand_like(original_noise).to(self.device) * 0.5
        theta = u + binary_reshaped.float() * 0.5
        adjustment = torch.erfinv(2 * theta[mask] - 1) * torch.sqrt(torch.tensor(2.0, device=self.device))
        noise=original_noise.clone().to(self.device)
        # print(f"noise.shape {noise.shape} before adjust")
        # print(noise)
        noise[mask] = adjustment
        # print(f"noise.shape {noise.shape} after adjust")
        # print(noise)
        # 高斯检验
        # Gaussian_test(noise,self.logger)
        return noise

    # 添加一个对noise进行处理的函数
    def _get_init_noise(self,noise: torch.Tensor, )->torch.Tensor:
        # if noise.size(1)==16:
        #     samples = noise.cpu().numpy().flatten()
        #     _, p_value = kstest(samples, 'norm', args=(0, 1))
        #     if np.var(samples) == 0 or p_value < 0.05: 
        #         self.logger.warning(f"[DPRWatermark]p_value {p_value}")
        #         self.logger.warning("[DPRWatermark]Embed watermark - Noise variance is 0 or p-value<0.05 using new random noise")
        #         noise=torch.randn(1,noise.size(1),noise.size(2),noise.size(3),device=self.device,dtype=noise.dtype)
        #         self.logger.warning(f"[DPRWatermark]Embed watermark - Using NEW random noise: {noise.shape}")
        #     else:
        #         self.logger.info(f"[DPRWatermark]Embed watermark - The noise is not empty and can be used directly")
        # elif self.latent_channels == 16 and noise.size(1) != 16:
        #     noise = torch.randn(1, self.latent_channels, noise.size(2), noise.size(3), device=self.device, dtype=noise.dtype)
        #     self.logger.warning(f"[DPRWatermark]Embed watermark - Using NEW random noise: {noise.shape}")
        # elif self.latent_channels == 4:
        #     # 先判断噪声是否是空值，如果是则重新生成
        #     samples = noise.cpu().numpy().flatten()
        #     _, p_value = kstest(samples, 'norm', args=(0, 1))
        #     if np.var(samples) == 0 or p_value < 0.05: 
        #         self.logger.warning(f"[DPRWatermark]p_value {p_value}")
        #         self.logger.warning("[DPRWatermark]Embed watermark - Noise variance is 0 or p-value<0.05 using new random noise")
        #         noise=torch.randn(1,noise.size(1),noise.size(2),noise.size(3),device=self.device,dtype=noise.dtype)
        #         self.logger.warning(f"[DPRWatermark]Embed watermark - Using NEW random noise: {noise.shape}")
        #     else:
        #         self.logger.info(f"[DPRWatermark]Embed watermark - The noise is not empty and can be used directly")
        # return noise
        if self.latent_channels == 16 and noise.size(1) != 16:
            noise = torch.randn(1, self.latent_channels, noise.size(2), noise.size(3), device=self.device, dtype=noise.dtype)
            self.logger.warning(f"Embed watermark - Using NEW random noise: {noise.shape}")
        elif self.latent_channels == 4:
            # 先判断噪声是否是空值，如果是则重新生成
            samples = noise.cpu().numpy().flatten()
            _, p_value = kstest(samples, 'norm', args=(0, 1))
            if np.var(samples) == 0 or p_value < 0.05: 
                self.logger.warning(f"p_value {p_value}")
                self.logger.warning("Embed watermark - Noise variance is 0 or p-value<0.05 using new random noise")
                noise=torch.randn(1,noise.size(1),noise.size(2),noise.size(3),device=self.device,dtype=noise.dtype)
                self.logger.warning(f"Embed watermark - Using NEW random noise: {noise.shape}")
            else:
                self.logger.info(f"Embed watermark - The noise is not empty and can be used directly")
        return noise

    def embed_watermark(self, noise: torch.Tensor, message: str, message_length: int, window_size: int, seed: int) -> [torch.Tensor,torch.Tensor]:
        """嵌入水印到噪声中"""
        self.logger.info(f"[DPRWatermark]Embed watermark - noise.dtype {noise.dtype}")
        self.logger.info(f"====================DPRW Watermark Embedding Begin====================")
        # 我们需要一个初始噪声，如果传入进来的初始噪声没法用，我们则生成一个，否则就用传入进来的初始噪声
        init_noise = self._get_init_noise(noise)
        self.logger.info(f"[DPRWatermark]Embed watermark - Noise shape: {init_noise.shape}")
        total_blocks = init_noise.numel() // (init_noise.shape[0] * window_size)
        self.logger.info(f"[DPRWatermark]Embed watermark - Total blocks: {total_blocks}")
        watermark = self._create_watermark(total_blocks, message, message_length)
        encrypted_bits = self._encrypt(watermark)
        binary = self._binarize_noise(init_noise)
        binary_embedded = self._embed_bits(binary, encrypted_bits, window_size)
        watermarked_noise = self._restore_noise(binary_embedded, init_noise.shape, seed,init_noise)
        self.logger.info(f"[DPRWatermark]watermarked_noise.shape {watermarked_noise.shape}")
        self.logger.info(f"====================DPRW Watermark Embedding End====================")
        # 返回水印的噪声和原始的噪声
        return watermarked_noise,init_noise

    # 如果不知道原水印，但知道它的长度也可以
    def extract_watermark(self, noise: torch.Tensor, message_length: int, window_size: int) -> tuple[str, str]:
        """从噪声中提取水印"""
        self.logger.info(f"====================DPRW Watermark Extract Begin====================")
        binary = self._binarize_noise(noise)
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        bits = windows.sum(dim=1) % 2
        bit_str = ''.join(bits.cpu().numpy().astype(str))
        byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decrypted = cipher.decryptor().update(byte_data) + cipher.decryptor().finalize()
        all_bits = ''.join(format(byte, '08b') for byte in decrypted)
        segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
        msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' for i in range(message_length))
        msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
        self.logger.info(f"====================DPRW Watermark Extract End====================")
        return msg_bin, msg

        # 水印准确性评估
    
    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str:str="") -> float:
        """计算位准确率"""
        self.logger.info(f"====================DPRW Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.logger.info(f"[DPRWatermark]Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"[DPRWatermark]Evaluation - Extracted binary: {extracted_bin}")
        self.logger.info(f"[DPRWatermark]Evaluation - Extracted binary length: {len(extracted_bin)}")
        if accuracy > 0.9:
            self.logger.info(f"Evaluation - Extracted message: {extracted_msg_str}")
        self.logger.info(f"[DPRWatermark]Evaluation - Bit accuracy: {accuracy}")
        self.logger.info(f"====================DPRW Watermark Evaluate End====================")
        return orig_bin,accuracy


class DPRLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "init_latent": ("LATENT",),
                "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
                "message": ("STRING", {"default": "lthero"}),
                "latent_channels": ("INT", {"default": 4, "min": 4, "max": 16,"step": 12}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 5}),
            }
        }

    RETURN_TYPES = ("LATENT","STRING","LATENT")
    FUNCTION = "create_watermarked_latents"
    CATEGORY = "DPRW/latent"

    def create_watermarked_latents(self, init_latent, use_seed, seed,  key, nonce, message,latent_channels, window_size):
        """创建带水印的潜在噪声"""
        if not isinstance(init_latent, dict) or "samples" not in init_latent:
            raise ValueError("init_latent must be a dictionary containing 'samples' key")
        init_noise = init_latent["samples"]
        # print(f"init_noise.shape {init_noise.shape}")
        dprw = DPRWatermark(key, nonce,latent_channels)
        if use_seed:
            set_random_seed(seed)
        message_length = len(message) * 8
        watermarked_noise,init_noise = dprw.embed_watermark(init_noise, message, message_length, window_size, seed)
        # print(f"watermarked_noise.shape {watermarked_noise.shape}")
        # print(f"init_noise.shape {init_noise.shape}")
        Message_length_str = f"Message length: {message_length}"
        return ({"samples": watermarked_noise},Message_length_str,{"samples": init_noise})
