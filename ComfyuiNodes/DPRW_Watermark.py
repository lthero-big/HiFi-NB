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

    def _create_watermark(self, total_blocks: int, message: str, message_length: int, BCH_AVAILABLE = False) -> bytes:
        """先BCH编码再重复：对消息进行BCH编码增强纠错能力，然后重复填满容量"""
        # 确定实际的比特长度
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        
        if not BCH_AVAILABLE:
            self.logger.warning("[DPRWatermark]BCH库未安装，使用原始重复编码")
            # Ensure bchlib is imported if you are going to call _create_watermark_original
            # import bchlib # Make sure this is available or handle appropriately
            return self._create_watermark_original(total_blocks, message, message_length)
        
        try:
            # BCH_AVAILABLE is true, so bchlib should be imported
            import bchlib # Ensure bchlib is imported here or at the top of the file

            # BCH编码参数定义为变量，方便修改
            BCH_PRIMITIVE_POLY = 285  # Primitive polynomial for GF(2^8) -> N=255. (e.g., 0x11D)
            BCH_N_BITS = 255          # BCH codeword length in bits (derived from m=8)
            
            # User's desired information bits per BCH block (BCH_K from original code)
            # This will be used to determine data_bytes for padding the input message
            TARGET_INFO_BITS = 45 # This was BCH_K in your original code

            # Calculate data_bytes based on TARGET_INFO_BITS
            # This is the number of bytes your input message will be padded/truncated to.
            data_bytes = (TARGET_INFO_BITS + 7) // 8 # e.g. (45+7)//8 = 6 bytes (48 bits)
            actual_info_bits_payload = data_bytes * 8 # 48 bits if TARGET_INFO_BITS=45

            max_t_for_payload = (BCH_N_BITS - actual_info_bits_payload) // 8
            
            BCH_T_CORRECTION_BITS = min(25, max_t_for_payload) # Use 25 or max possible if less
            if BCH_T_CORRECTION_BITS < 1:
                 self.logger.error(f"[DPRWatermark] Cannot achieve positive error correction for payload {actual_info_bits_payload} bits.")
                 raise ValueError("BCH parameters lead to no error correction capability.")

            self.logger.info(f"[DPRWatermark]Create watermark - BCH Params: N={BCH_N_BITS}, Target Info Bits (for padding msg)={TARGET_INFO_BITS} ({actual_info_bits_payload} actual bits after byte padding)")
            self.logger.info(f"[DPRWatermark]Create watermark - Chosen BCH Primitive Poly={BCH_PRIMITIVE_POLY} (for m=8)")
            self.logger.info(f"[DPRWatermark]Create watermark - Chosen BCH T (error correction bits)={BCH_T_CORRECTION_BITS}")
            
            # 创建BCH编码器
            # bchlib.BCH(primitive_polynomial, t_error_correction_bits)
            bch = bchlib.BCH(BCH_PRIMITIVE_POLY, BCH_T_CORRECTION_BITS)
            
            
            if data_bytes != bch.k_bytes:
                self.logger.warning(f"[DPRWatermark] Mismatch: Padded message data_bytes ({data_bytes}) vs BCH lib expected k_bytes ({bch.k_bytes}).")
                self.logger.warning(f"[DPRWatermark] Lib k_bits: {bch.k_bits}, Lib ecc_bits: {bch.ecc_bits}, Lib n_bits: {bch.n_bits}")


            # 将消息转为字节
            msg_bytes = message.encode('utf-8')
            
            self.logger.info(f"[DPRWatermark]Create watermark - 每个BCH块可容纳信息字节数 (based on TARGET_INFO_BITS): {data_bytes}")
            
            # 填充/截断消息到适合的长度 (data_bytes, which should match bch.k_bytes)
            padded_msg = msg_bytes.ljust(data_bytes, b'\x00')[:data_bytes]
            
            self.logger.info(f"[DPRWatermark]Create watermark - 原始消息: '{message}'")
            self.logger.info(f"[DPRWatermark]Create watermark - 消息长度: {len(msg_bytes)*8}位 -> 填充至{len(padded_msg)*8}位 for BCH input.")
            
            # BCH编码
            ecc_payload = bch.encode(padded_msg) # ecc_payload contains the error correction bits
            
            # 编码结果是校验位，需要将原始数据和校验位组合
            bch_block_bytes = bytes(padded_msg) + bytes(ecc_payload) # Ensure both are bytes
            
            
            self.logger.info(f"[DPRWatermark]Create watermark - BCH Encoded Block: {len(padded_msg)*8} data bits + {bch.ecc_bits} ECC bits = {len(padded_msg)*8 + bch.ecc_bits} total bits in logical block.")
            self.logger.info(f"[DPRWatermark]Create watermark - BCH Encoded Block on disk (bytes): {len(bch_block_bytes)} bytes.")


            current_bch_block_byte_length = len(bch_block_bytes)


            # 计算需要重复的次数
            total_bytes_capacity = total_blocks // 8 # total_blocks is number of bits watermark can occupy
            if current_bch_block_byte_length == 0:
                self.logger.error("[DPRWatermark] BCH block byte length is 0. Cannot proceed.")
                raise ValueError("BCH block length is zero after encoding.")
            
            repeats = total_bytes_capacity // current_bch_block_byte_length
            
            self.logger.info(f"[DPRWatermark]Create watermark - BCH block size for repetition: {current_bch_block_byte_length} bytes")
            self.logger.info(f"[DPRWatermark]Create watermark - Repetition times: {repeats}")
            
            # 创建最终的水印：BCH编码块的重复
            result = bch_block_bytes * repeats
            
            # 处理可能的剩余空间
            remaining_bytes = total_bytes_capacity - len(result)
            if remaining_bytes > 0:
                result += bch_block_bytes[:remaining_bytes]
            
            return result
                
        except ImportError: # Catch if bchlib was not available after all
            self.logger.warning("[DPRWatermark]BCH库(bchlib)导入失败或不可用，使用原始重复编码")
            return self._create_watermark_original(total_blocks, message, message_length)
        except Exception as e:
            import traceback # Import for full traceback
            self.logger.warning(f"[DPRWatermark]BCH编码失败: {e}: {traceback.format_exc()}")
            return self._create_watermark_original(total_blocks, message, message_length)



    def _create_watermark_original(self, total_blocks: int, message: str, message_length: int) -> bytes:
        """原始的水印生成方法，使用简单重复"""
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        self.logger.info(f"[DPRWatermark]Create watermark - Message: {message} ")
        self.logger.info(f"[DPRWatermark]Create watermark - Message Length: {message_length}, total bits: {total_blocks}")
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

    def _quantile_resample_v2(self, noise: torch.Tensor, mask: torch.Tensor, 
                            binary_reshaped: torch.Tensor, original_noise: torch.Tensor) -> torch.Tensor:
        """使用 torch.quantile 进行基于经验分布的重采样"""
        
        if mask.sum() == 0:
            return noise
        
        # 1. 展平原始噪声数据
        original_flat = original_noise.flatten().to(self.device)
        
        # 2. 计算中位数作为分界点
        median_value = torch.quantile(original_flat, 0.5)
        
        self.logger.info(f"[DPRWatermark] 原始数据中位数: {median_value:.4f}")
        
        # 3. 生成均匀随机数
        num_samples = mask.sum().item()
        u_values = torch.rand(num_samples, device=self.device)
        target_bits = binary_reshaped[mask].float()
        
        # 4. 根据目标位值调整采样分位数
        # bit=0: 从 [0, 0.5) 分位数范围采样
        # bit=1: 从 [0.5, 1.0] 分位数范围采样
        quantile_values = u_values * 0.5 + target_bits * 0.5
        
        # 5. 使用 torch.quantile 直接计算对应的值
        # 确保 quantile_values 在有效范围内
        quantile_values = torch.clamp(quantile_values, 0.0001, 0.9999)  # 避免边界值
        
        # 使用 torch.quantile 计算对应的数值
        sampled_values = torch.quantile(original_flat, quantile_values)
        
        # 6. 赋值给需要调整的位置
        noise[mask] = sampled_values
        
        return noise

    def _quantile_resample(self, noise: torch.Tensor, mask: torch.Tensor, 
                         binary_reshaped: torch.Tensor, original_noise: torch.Tensor) -> torch.Tensor:
        """使用 torch.quantile 的高级版本 - 更精确的分位数控制"""
        
        if mask.sum() == 0:
            return noise
        
        # 1. 展平原始噪声数据
        original_flat = original_noise.flatten().to(self.device)
        
        # 2. 获取需要调整的位置和目标位值
        target_bits = binary_reshaped[mask]
        num_samples = mask.sum().item()
        
        # 3. 生成分层采样的分位数
        sampled_values = torch.zeros(num_samples, device=self.device, dtype=noise.dtype)
    
        u_values = torch.rand(num_samples, device=self.device)
        
        # 根据目标位值调整采样分位数
        # bit=0: 从 [0, 0.5) 分位数范围采样
        # bit=1: 从 [0.5, 1.0] 分位数范围采样
        quantile_values = u_values * 0.5 + target_bits.float() * 0.5
        
        # 确保分位数在有效范围内
        quantile_values = torch.clamp(quantile_values, 0.0001, 0.9999)
        
        # 使用 torch.quantile 计算对应值
        sampled_values = torch.quantile(original_flat, quantile_values)
        
        # 5. 赋值
        noise[mask] = sampled_values
        
        return noise


    def _restore_noise(self, binary: torch.Tensor, shape: tuple, seed: int,original_noise:torch.Tensor, resampling_method: str = "quantile") -> torch.Tensor:
        """还原成连续值噪声"""
        original_noise = original_noise.to(self.device)
        binary_reshaped = binary.view(original_noise.shape).to(self.device)
        # 根据重采样方法选择合适的二值化策略来计算 original_binary
        if resampling_method == "quantile":
            # 对于分位数重采样，使用中位数作为阈值
            threshold = torch.median(original_noise.flatten())
            original_binary = (original_noise > threshold).to(torch.uint8).to(self.device)
            self.logger.info(f"[DPRWatermark] 分位数方法使用中位数阈值: {threshold:.4f}")
        elif resampling_method == "inverse":
            # 对于符号反转，也使用中位数更稳健
            threshold = torch.median(original_noise.flatten())
            original_binary = (original_noise > threshold).to(torch.uint8).to(self.device)
            self.logger.info(f"[DPRWatermark] 符号反转方法使用中位数阈值: {threshold:.4f}")
        else:
            # 对于标准重采样方法，使用原来的 sigmoid 方法
            original_binary = (torch.sigmoid(original_noise) > 0.5).to(torch.uint8).to(self.device)
            self.logger.info(f"[DPRWatermark] 标准重采样方法使用sigmoid阈值")
        
        mask = binary_reshaped != original_binary
        noise=original_noise.clone().to(self.device)
        if resampling_method == "inverse":
            # 新方案：符号反转
            # noise[mask] = original_noise[mask] * (-1)
            noise[mask] = original_noise.to(self.device)[mask] * (-1)
            self.logger.info(f"[DPRWatermark] 使用符号反转方案，反转了 {mask.sum().item()} 个位置, 占总 {mask.numel()} 个位置的 {mask.sum().item() / mask.numel() * 100:.2f}%")
        elif resampling_method == 'quantile':
            # 新方案：分位数重采样
            noise = self._quantile_resample(noise, mask, binary_reshaped, original_noise)
            self.logger.info(f"[DPRWatermark] 使用分位数重采样方案，调整了 {mask.sum().item()} 个位置, 占{mask.numel()} 中的 {mask.sum().item() / mask.numel() * 100:.2f}%")
        elif resampling_method == 'resampling':
            # 原方案：重采样
            u = torch.rand_like(original_noise).to(self.device) * 0.5
            theta = u + binary_reshaped.float() * 0.5
            adjustment = torch.erfinv(2 * theta[mask] - 1) * torch.sqrt(torch.tensor(2.0, device=self.device))
            noise[mask] = adjustment
            self.logger.info(f"[DPRWatermark] 使用重采样方案，调整了 {mask.sum().item()} 个位置, 占{mask.numel()} 中的 {mask.sum().item() / mask.numel() * 100:.2f}%")
        else:
            raise ValueError(f"[DPRWatermark] Unsupported resampling method: {resampling_method}. Choose 'inverse', 'quantile', or 'resampling'.")

        # set_random_seed(seed)
        # noise = torch.randn(shape, device=self.device)
        # binary_reshaped = binary.view(shape[1:]).to(self.device)


        # u = torch.rand_like(original_noise).to(self.device) * 0.5
        # theta = u + binary_reshaped.float() * 0.5
        # adjustment = torch.erfinv(2 * theta[mask] - 1) * torch.sqrt(torch.tensor(2.0, device=self.device))
        # noise=original_noise.clone().to(self.device)
        # noise[mask] = adjustment


        # binary_reshaped = binary.view(original_noise.shape).to(self.device)
        # original_binary = (torch.sigmoid(original_noise) > 0.5).to(torch.uint8).to(self.device)
        # mask = binary_reshaped != original_binary
        return noise

    # 添加一个对noise进行处理的函数
    def _get_init_noise(self,noise: torch.Tensor, )->torch.Tensor:
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

    def embed_watermark(self, noise: torch.Tensor, message: str, message_length: int, window_size: int, seed: int, resampling_method:  str = "quantile") -> [torch.Tensor,torch.Tensor]:
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
        watermarked_noise = self._restore_noise(binary_embedded, init_noise.shape, seed,init_noise, resampling_method)
        self.logger.info(f"[DPRWatermark]watermarked_noise.shape {watermarked_noise.shape}")
        self.logger.info(f"====================DPRW Watermark Embedding End====================")
        # 返回水印的噪声和原始的噪声
        return watermarked_noise,init_noise

    # quantile / resampling
    def extract_watermark(self, noise: torch.Tensor, message_length: int = 0, window_size: int = 1, BCH_AVAILABLE = False, resampling_method = "quantile") -> tuple[str, str, list]:
        """从噪声中提取基于BCH编码的水印"""
        self.logger.info(f"====================DPRW Watermark Extract Begin====================")
        
        # 1. 从噪声中提取二进制序列
        binary = self._binarize_noise(noise)
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        bits = windows.sum(dim=1) % 2
        bit_str = ''.join(bits.cpu().numpy().astype(str))
        
        # 2. 转换为字节
        byte_data = bytearray()
        for i in range(0, len(bit_str) - 7, 8):
            byte_data.append(int(bit_str[i:i + 8], 2))
        
        # 3. 解密数据
        try:
            cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
            decrypted = cipher.decryptor().update(bytes(byte_data)) + cipher.decryptor().finalize()
        except Exception as e:
            self.logger.error(f"[DPRWatermark] 解密失败: {e}")
            return "", "", []
        
        # 如果BCH不可用，直接使用原始方法
        if not BCH_AVAILABLE:
            self.logger.warning("[DPRWatermark] BCH库未安装，使用原始方法")
            return self.extract_watermark_original(noise, message_length, window_size, resampling_method)
        
        # 4. 尝试BCH解码
        try:
            # BCH编码参数，与嵌入时保持一致
            BCH_N = 255  # BCH码字长度
            BCH_K = 45   # BCH信息长度
            BCH_T = 45   # BCH纠错能力
            
            # 计算各种字节长度
            bch_block_bytes_expected = (BCH_N + 7) // 8  # BCH块的字节长度，向上取整
            info_bytes = (BCH_K + 7) // 8  # 信息字节长度，向上取整
            
            # 如果数据太短，回退到原始方法
            if len(decrypted) < bch_block_bytes_expected:
                self.logger.warning(f"[DPRWatermark] 解密数据太短: {len(decrypted)}字节，使用原始方法")
                return self.extract_watermark_original(noise, message_length, window_size, resampling_method)
            
            # 创建BCH解码器 - 使用BCH_N而不是8219
            bch = bchlib.BCH(BCH_N, BCH_T)
            
            # 提取多个BCH块
            blocks = []
            for i in range(0, len(decrypted), bch_block_bytes_expected):
                if i + bch_block_bytes_expected <= len(decrypted):
                    blocks.append(decrypted[i:i+bch_block_bytes_expected])
            
            self.logger.info(f"[DPRWatermark] 检测到{len(blocks)}个BCH块")
            
            # 解码BCH块
            decoded_results = []
            for block_idx, block in enumerate(blocks[:50]):  # 最多处理50个块
                try:
                    # 分离数据和校验位
                    data = block[:info_bytes]
                    ecc = block[info_bytes:bch_block_bytes_expected]
                    
                    # 尝试解码
                    bitflips = bch.decode(data, ecc)
                    if bitflips != -1:  # -1表示无法纠正
                        # 成功解码
                        self.logger.info(f"[DPRWatermark] 块{block_idx}解码成功，纠正了{bitflips}位错误")
                        
                        # 移除填充零
                        decoded = data.rstrip(b'\x00')
                        
                        if decoded:
                            try:
                                # 尝试解码为文本
                                text = decoded.decode('utf-8', errors='replace')
                                confidence = 1.0 - (bitflips / (BCH_T * 2))  # 置信度计算
                                decoded_results.append((text, decoded, confidence, bitflips))
                            except Exception as text_err:
                                self.logger.warning(f"[DPRWatermark] 文本解码失败: {text_err}")
                except Exception as block_err:
                    self.logger.warning(f"[DPRWatermark] 块{block_idx}解码失败: {block_err}")
            
            # 如果有成功解码的结果
            if decoded_results:
                # 按修正的错误位数排序，错误最少的最可靠
                decoded_results.sort(key=lambda x: x[3])
                
                # 统计解码结果
                text_counts = {}
                for text, _, _, _ in decoded_results:
                    text_counts[text] = text_counts.get(text, 0) + 1
                
                # 按出现频率排序
                sorted_results = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)
                most_common_text = sorted_results[0][0]
                frequency = sorted_results[0][1]
                
                # 找到对应的原始二进制
                for text, decoded, conf, _ in decoded_results:
                    if text == most_common_text:
                        msg_bin = ''.join(format(b, '08b') for b in decoded)
                        msg_text = text
                        break
                else:
                    msg_bin = ''.join(format(b, '08b') for b in decoded_results[0][1])
                    msg_text = decoded_results[0][0]
                
                self.logger.info(f"[DPRWatermark] 成功提取消息: '{msg_text}'，出现频率: {frequency}/{len(decoded_results)}")
                self.logger.info(f"====================DPRW Watermark Extract End====================")
                return msg_bin, msg_text, blocks
            
            # 5. 尝试使用多数投票解码
            self.logger.warning("[DPRWatermark] BCH解码失败，尝试使用多数投票")
            
            # 处理消息长度
            if message_length > 0:
                # 如果提供了消息长度，直接使用
                data_bytes = (message_length + 7) // 8
            else:
                # 否则使用BCH信息长度
                data_bytes = info_bytes
            
            # 提取每个块的信息部分
            data_blocks = [block[:data_bytes] for block in blocks if len(block) >= data_bytes]
            
            if data_blocks:
                # 按字节进行多数投票
                voted_data = bytearray()
                min_len = min(len(block) for block in data_blocks)
                
                for i in range(min_len):
                    byte_counts = {}
                    for block in data_blocks:
                        byte_counts[block[i]] = byte_counts.get(block[i], 0) + 1
                    
                    # 找出出现次数最多的字节
                    most_common = max(byte_counts.items(), key=lambda x: x[1])
                    voted_data.append(most_common[0])
                
                # 移除填充零
                while voted_data and voted_data[-1] == 0:
                    voted_data.pop()
                    
                # 尝试解码为文本
                try:
                    msg_text = voted_data.decode('utf-8', errors='replace')
                    msg_bin = ''.join(format(b, '08b') for b in voted_data)
                    
                    self.logger.info(f"[DPRWatermark] 多数投票提取成功: '{msg_text}'")
                    self.logger.info(f"====================DPRW Watermark Extract End====================")
                    return msg_bin, msg_text, blocks
                except Exception as e:
                    self.logger.warning(f"[DPRWatermark] 投票文本解码失败: {e}")
                    return self.extract_watermark_original(noise, message_length, window_size, resampling_method)
        
        except ImportError:
            self.logger.warning("[DPRWatermark] BCH库未安装，使用原始方法")
            return self.extract_watermark_original(noise, message_length, window_size, resampling_method)
        except Exception as e:
            self.logger.warning(f"[DPRWatermark] BCH解码过程失败: {e}: {traceback.format_exc()}")
            return self.extract_watermark_original(noise, message_length, window_size, resampling_method)
        # 回退到原始方法
        self.logger.info("[DPRWatermark] 回退到原始提取方法")
        return self.extract_watermark_original(noise, message_length, window_size, resampling_method)


    def _binarize_noise_adaptive(self, noise: torch.Tensor, method: str = "median") -> torch.Tensor:
        """自适应噪声二值化，根据噪声分布特征进行二值化"""
        noise_flat = noise.flatten().to(self.device)
        
        if method == "median":
            # 使用中位数作为分界点
            threshold = torch.median(noise_flat)
            self.logger.info(f"[DPRWatermark] 使用中位数阈值进行二值化: {threshold:.4f}")
        elif method == "mean":
            # 使用均值作为分界点
            threshold = torch.mean(noise_flat)
            self.logger.info(f"[DPRWatermark] 使用均值阈值进行二值化: {threshold:.4f}")
        elif method == "quantile":
            # 使用50%分位数（等同于中位数，但更明确）
            threshold = torch.quantile(noise_flat, 0.5)
            self.logger.info(f"[DPRWatermark] 使用50%分位数阈值进行二值化: {threshold:.4f}")
        else:
            # 默认使用sigmoid方法（适用于标准正态分布）
            self.logger.info("[DPRWatermark] 使用标准sigmoid阈值进行二值化")
            return (torch.sigmoid(noise.to(self.device)) > 0.5).to(torch.uint8).flatten()
        
        # 基于阈值进行二值化
        return (noise_flat > threshold).to(torch.uint8)

    def extract_watermark_adaptive(self, noise: torch.Tensor, message_length: int, window_size: int, 
                                resampling_method: str = "quantile") -> tuple[str, str, list]:
        """自适应水印提取，根据重采样方法选择合适的二值化策略"""
        self.logger.info(f"====================DPRW Watermark Extract Begin====================")
        self.logger.info(f"[DPRWatermark] 提取方法: {resampling_method}")
        
        # 根据重采样方法选择二值化策略
        if resampling_method == "quantile":
            print("resampling_method == quantile", "self._binarize_noise_adaptive(noise, method=median)")
            binary = self._binarize_noise_adaptive(noise, method="median")
        elif resampling_method == "quantile2":
            print("resampling_method == quantile2", "self._binarize_noise_adaptive(noise, method=quantile)")
            binary = self._binarize_noise_adaptive(noise, method="quantile")
        elif resampling_method == "inverse":
            print("resampling_method == inverse", "self._binarize_noise_adaptive(noise, method=median)")
            # 符号反转方法可能改变了分布的对称性，使用中位数更稳健
            binary = self._binarize_noise_adaptive(noise, method="median")
        elif resampling_method == "resampling":
            print("resampling_method == resampling", "self._binarize_noise_adaptive(noise, method=resampling)")
            # 标准重采样方法，使用原始的sigmoid方法
            binary = self._binarize_noise_adaptive(noise, method="resampling")
        else:
            self.logger.error(f"[DPRWatermark] 不支持的重采样方法: {resampling_method}")
            return "", "", []
        # 提取水印位
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        bits = windows.sum(dim=1) % 2
        bit_str = ''.join(bits.cpu().numpy().astype(str))
        
        self.logger.info(f"[DPRWatermark] 提取到 {len(bit_str)} 位二进制数据")
        
        # 解密和解码
        try:
            byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
            cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
            decrypted = cipher.decryptor().update(byte_data) + cipher.decryptor().finalize()
            all_bits = ''.join(format(byte, '08b') for byte in decrypted)
            
            # 分段处理
            segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
            
            # 多数投票
            msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' for i in range(message_length))
            msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
            
            self.logger.info(f"[DPRWatermark] 成功提取消息: '{msg}'")
            self.logger.info(f"====================DPRW Watermark Extract End====================")
            return msg_bin, msg, segments
            
        except Exception as e:
            self.logger.error(f"[DPRWatermark] 提取过程中出错: {e}")
            self.logger.info(f"====================DPRW Watermark Extract End====================")
            return "", "", []

    def extract_watermark_with_distribution_analysis(self, noise: torch.Tensor, message_length: int, window_size: int) -> tuple[str, str, list]:
        """基于分布分析的水印提取"""
        self.logger.info(f"====================DPRW Watermark Extract Begin====================")
        
        # 分析噪声分布特征
        noise_flat = noise.flatten().to(self.device)
        mean_val = torch.mean(noise_flat)
        std_val = torch.std(noise_flat)
        median_val = torch.median(noise_flat)
        
        # 计算偏度（简单估计）
        centered = noise_flat - mean_val
        skewness = torch.mean(centered**3) / (std_val**3)
        
        self.logger.info(f"[DPRWatermark] 噪声分布分析:")
        self.logger.info(f"  均值: {mean_val:.4f}")
        self.logger.info(f"  标准差: {std_val:.4f}")
        self.logger.info(f"  中位数: {median_val:.4f}")
        self.logger.info(f"  偏度: {skewness:.4f}")
        
        # 根据分布特征选择二值化方法
        if abs(skewness) > 0.5:
            # 分布有明显偏斜，使用中位数
            threshold = median_val
            method = "median"
        elif abs(mean_val - median_val) > 0.1 * std_val:
            # 均值和中位数差异较大，使用中位数
            threshold = median_val
            method = "median"
        else:
            # 分布相对对称，可以使用均值
            threshold = mean_val
            method = "mean"
        
        self.logger.info(f"[DPRWatermark] 选择二值化方法: {method}, 阈值: {threshold:.4f}")
        
        # 进行二值化
        binary = (noise_flat > threshold).to(torch.uint8)
        
        # 提取水印位
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        bits = windows.sum(dim=1) % 2
        bit_str = ''.join(bits.cpu().numpy().astype(str))
        
        # 解密和解码过程与原方法相同
        try:
            byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
            cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
            decrypted = cipher.decryptor().update(byte_data) + cipher.decryptor().finalize()
            all_bits = ''.join(format(byte, '08b') for byte in decrypted)
            segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
            msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' for i in range(message_length))
            msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
            
            self.logger.info(f"[DPRWatermark] 成功提取消息: '{msg}'")
            self.logger.info(f"====================DPRW Watermark Extract End====================")
            return msg_bin, msg, segments
            
        except Exception as e:
            self.logger.error(f"[DPRWatermark] 提取过程中出错: {e}")
            self.logger.info(f"====================DPRW Watermark Extract End====================")
            return "", "", []
    
    # 修改原有的提取方法，增加重采样方法参数
    def extract_watermark_original(self, noise: torch.Tensor, message_length: int, window_size: int, 
                                resampling_method: str = "quantile") -> tuple[str, str, list]:
        """从噪声中提取水印（改进版）"""

        print("extract_watermark_original", "resampling_method:", resampling_method)
        return self.extract_watermark_adaptive(noise, message_length, window_size, resampling_method)

    # # 如果不知道原水印，但知道它的长度也可以
    # def extract_watermark_original(self, noise: torch.Tensor, message_length: int, window_size: int) -> tuple[str, str]:
    #     """从噪声中提取水印"""
    #     self.logger.info(f"====================DPRW Watermark Extract Begin====================")
    #     binary = self._binarize_noise(noise)
    #     num_windows = len(binary) // window_size
    #     windows = binary[:num_windows * window_size].view(num_windows, window_size)
    #     bits = windows.sum(dim=1) % 2
    #     bit_str = ''.join(bits.cpu().numpy().astype(str))
    #     byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
    #     cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
    #     decrypted = cipher.decryptor().update(byte_data) + cipher.decryptor().finalize()
    #     all_bits = ''.join(format(byte, '08b') for byte in decrypted)
    #     segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
    #     msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' for i in range(message_length))
    #     msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
    #     self.logger.info(f"====================DPRW Watermark Extract End====================")
    #     return msg_bin, msg, segments

    #     # 水印准确性评估
    
    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str:str="", extracted_msg_bin_segments="") -> float:
        """计算位准确率"""
        self.logger.info(f"====================DPRW Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        accuracyOfsegments = []
        if extracted_msg_bin_segments is not None and len(extracted_msg_bin_segments) > 0:
            # 如果提供了提取的二进制段，分别计算每个段的准确率
            for segment in extracted_msg_bin_segments:
                min_len = min(len(orig_bin), len(segment))
                orig_bin, segment = orig_bin[:min_len], segment[:min_len]
                accuracy = sum(a == b for a, b in zip(orig_bin, segment)) / min_len
                accuracyOfsegments.append(accuracy)
                # self.logger.info(f"[DPRWatermark]Evaluation - Segment accuracy: {accuracy}")
            self.logger.info(f"[DPRWatermark]Evaluation - Average Segment accuracy: {sum(accuracyOfsegments)/len(accuracyOfsegments)}, length of segments: {len(extracted_msg_bin_segments)}")
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
                "resampling_method": ("STRING", {"default": "quantile", "choices": ["inverse", "quantile", "resampling"]}),
            }
        }

    RETURN_TYPES = ("LATENT","STRING","LATENT")
    FUNCTION = "create_watermarked_latents"
    CATEGORY = "DPRW/latent"

    def create_watermarked_latents(self, init_latent, use_seed, seed,  key, nonce, message,latent_channels, window_size, resampling_method):
        """创建带水印的潜在噪声"""
        if not isinstance(init_latent, dict) or "samples" not in init_latent:
            raise ValueError("init_latent must be a dictionary containing 'samples' key")
        init_noise = init_latent["samples"]
        # print(f"init_noise.shape {init_noise.shape}")
        dprw = DPRWatermark(key, nonce,latent_channels)
        if use_seed:
            set_random_seed(seed)
        message_length = len(message) * 8
        watermarked_noise,init_noise = dprw.embed_watermark(init_noise, message, message_length, window_size, seed, resampling_method)
       
        # print(f"watermarked_noise.shape {watermarked_noise.shape}")
        # print(f"init_noise.shape {init_noise.shape}")
        Message_length_str = f"Message length: {message_length}"
        watermarked_noise_compact = {"samples": watermarked_noise}
        # save_latent_data(init_latent, "init_noise_fromDPRWLatent")
        save_latent_data(watermarked_noise_compact, "watermarked_noise_fromDPRWLatent")
        return ({"samples": watermarked_noise},Message_length_str,{"samples": init_noise})
