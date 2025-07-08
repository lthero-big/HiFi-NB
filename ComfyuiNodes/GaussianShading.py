from .DPRW_utils import *

class GSWatermark:
    """
    GSWatermark 类：基于 Gaussian Shading 算法生成水印噪声
    """
    def __init__(self, key_hex: str, nonce_hex: str, device: str = "cuda", use_seed: bool = True, seed: int = 42, 
    watermark_length_bits:int=64,latent_channels: int = 4,log_dir: str = './logs'):
        self.key = validate_hex(key_hex, 64, os.urandom(32))
        self.nonce = validate_hex(nonce_hex, 32, os.urandom(16))
        self.device = device
        self.use_seed = use_seed
        self.seed = seed
        self.watermark_length=watermark_length_bits
        self.channels = latent_channels
        self.logger = Loggers.get_logger(log_dir)
        self.logger.info(f"====================GS Watermark Initialized Begin====================")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")
        self.logger.info(f"Initialized - Channels: {self.channels}")
        self.logger.info(f"====================GS Watermark Initialized End====================")

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        """生成水印字节"""
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        self.logger.info(f"Create watermark - Message: {message}")
        self.logger.info(f"Create watermark - Message Length: {message_length}")
        self.logger.info(f"Create watermark - Watermark repeats: {repeats} times")
        return padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)

    def _encrypt(self, watermark: bytes) -> str:
        """加密水印并转换为二进制字符串"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _embed_watermark(self,encrypted_watermark_bits:str, latent_height, latent_width,window_size:int=1)->torch.Tensor:
        Z_s_T_array = torch.zeros((self.channels, latent_height, latent_width), dtype=torch.float32, device=self.device)
        index = 0
        if self.use_seed:
            rng = np.random.RandomState(seed=self.seed)
        for i in range(0, len(encrypted_watermark_bits), window_size):
            window = encrypted_watermark_bits[i:i+window_size]
            y = int(window, 2)
            if not self.use_seed:
                u = np.random.uniform(0, 1)
            else:
                u = rng.uniform(0, 1)
            z_s_T = norm.ppf((u + y) / (2 ** window_size))
            channel_index = index // (latent_height * latent_width)
            h_index = (index // latent_width) % latent_height
            w_index = index % latent_width
            Z_s_T_array[channel_index, h_index, w_index] = z_s_T
            index += 1
            if index >= self.channels * latent_height * latent_width:
                break
        # Gaussian_test(Z_s_T_array,self.logger)
        return Z_s_T_array

    def init_noise(self, message: str, Image_width: int, Image_height: int,window_size:int=1) -> torch.Tensor:
        self.message=message
        latent_width= Image_width // 8
        latent_height = Image_height // 8
        total_blocks = self.channels * latent_width * latent_height

        watermark_message = self._create_watermark(total_blocks, message, self.watermark_length)
        encrypted_watermark_bits = self._encrypt(watermark_message)

        return self._embed_watermark(encrypted_watermark_bits,latent_width,latent_height,window_size)
    
    def extract_watermark(self,reversed_latents,message_length,window_size):
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Reconstruct m from reversed_latents
        reconstructed_m_bits = []
        for z_s_T_value in np.nditer(reversed_latents):
            y_reconstructed = norm.cdf(z_s_T_value) * 2**window_size
            reconstructed_m_bits.append(int(y_reconstructed))

        m_reconstructed_bytes = bytes(
            int(''.join(str(int(bit) % 2) for bit in reconstructed_m_bits[i:i+8]), 2)
            for i in range(0, len(reconstructed_m_bits), 8)
        )
        s_d_reconstructed = decryptor.update(m_reconstructed_bytes) + decryptor.finalize()
        bits_list = ['{:08b}'.format(byte) for byte in s_d_reconstructed]
        all_bits = ''.join(bits_list)

        segments = [seg for seg in (all_bits[i:i + message_length] for i in range(0, len(all_bits), message_length)) if len(seg) == message_length]
        reconstructed_message_bin = ''
        for i in range(message_length):
            count_1 = sum(segment[i] == '1' for segment in segments)
            reconstructed_message_bin += '1' if count_1 > len(segments) / 2 else '0'
        msg = bytes(int(reconstructed_message_bin[i:i + 8], 2) for i in range(0, len(reconstructed_message_bin), 8)).decode('utf-8', errors='replace')
        return reconstructed_message_bin,msg
    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str:str="") -> float:
        """计算位准确率"""
        self.logger.info(f"====================GS Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.logger.info(f"Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"Evaluation - Extracted binary: {extracted_bin}")
        self.logger.info(f"Evaluation - Extracted binary length: {len(extracted_bin)}")
        if accuracy > 0.9:
            self.logger.info(f"Evaluation - Extracted message: {extracted_msg_str}")
        self.logger.info(f"Evaluation - Bit accuracy: {accuracy}")
        self.logger.info(f"====================GS Watermark Evaluate End====================")
        return orig_bin,accuracy

class GSLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            "channels": ("INT", {"default": 4, "min": 4, "max": 16,"step":12}),
            "Image_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "Image_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
            "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
            "message": ("STRING", {"default": "lthero"}),
            "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
         }}
    RETURN_TYPES = ("LATENT","STRING")
    FUNCTION = "create_gs_latents"
    CATEGORY = "DPRW/latent"
    
    def create_gs_latents(self, key, nonce, message, batch_size, use_seed, seed, Image_width, Image_height,channels,window_size):
        device = "cuda"
        message_length=len(message)*8
        gs_watermark = GSWatermark(key_hex=key, nonce_hex=nonce, device=device, use_seed=bool(use_seed), seed=seed, 
                            watermark_length_bits=message_length,latent_channels=channels)
        latent_list = [gs_watermark.init_noise(message, Image_width, Image_height,window_size) for _ in range(batch_size)]
        latent = torch.stack(latent_list)
        Message_length_str = f"Message length: {message_length}"
        return ({"samples": latent},Message_length_str)
