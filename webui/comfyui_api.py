from tqdm import tqdm
import time
import sys
from datetime import datetime
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import hashlib
import pickle
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
# TOKEN is stored in the file `./PASSWORD`, or you can obtain it from the command line window when ComfyUI starts.
# It will appear like this:
# For direct API calls, token=$2b$12$qUfJfV942nrMiX77QRVgIuDk1.oyXBP7FYrXVEBqouTk.uP/hiqAK
TOKEN = "$2b$12$6kIVKLcQ9lh9Qcy8GwLA3Ormbr0lnExPUimaT49PGH9mlbNaOBLaG"
# 添加一个标志文件来标记是否是首次运行
FIRST_RUN_FLAG = Path("comfyui_cache/.first_run")
# 生成+提取工作流文件
aio_workflow_file = "workflow_api_jsons/Demo-[HiFi-NB]Flux.json"
# 仅提取工作流文件
extraction_workflow_file = "workflow_api_jsons/Demo-[DPRW]Flux_extraction_FromUPLoad_and_DDIM_inversion.json"

def generate_difference( img1, img2, scale_factor=1.0,use_color=False):
    # Convert PIL images to NumPy arrays (OpenCV format)
    if not isinstance(img1, np.ndarray):
        img1_np = np.array(img1.convert('RGB'))
    else:
        img1_np = img1
    if not isinstance(img2, np.ndarray):
        img2_np = np.array(img2.convert('RGB'))
    else:
        img2_np = img2
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
    # Convert to grayscale
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    if use_color:
        # Apply colormap for better visualization (optional)
        diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    # Convert back to RGB for PIL
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    return Image.fromarray(diff_rgb)

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt?token={}".format(server_address, TOKEN), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}&token={}".format(server_address, url_values, TOKEN)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}?token={}".format(server_address, prompt_id, TOKEN)) as response:
        return json.loads(response.read())

def get_workflow_hash(workflow_json, custom_settings=None):
    """为工作流和自定义设置生成唯一哈希值"""
    # 创建一个可序列化的工作流副本
    workflow_copy = workflow_json.copy()
    
    # 应用自定义设置到副本
    if custom_settings:
        for node_id, settings in custom_settings.items():
            if node_id in workflow_copy:
                for key, value in settings.items():
                    if 'inputs' in workflow_copy[node_id] and key in workflow_copy[node_id]['inputs']:
                        workflow_copy[node_id]['inputs'][key] = value
    
    # 序列化工作流并生成哈希
    serialized = pickle.dumps(workflow_copy)
    return hashlib.sha256(serialized).hexdigest()

def extract_text_outputs(history, target_node_titles=None):
    """从历史记录中提取文本输出，支持按节点标题匹配"""
    text_outputs = {}
    
    # 获取实际的 prompt 字典 (history['prompt'] 是列表，第三个元素才是节点字典)
    if isinstance(history['prompt'], list) and len(history['prompt']) >= 3:
        prompt_dict = history['prompt'][2]  # 第三个元素是节点字典
    else:
        print("警告: 历史记录格式不正确，无法正确解析节点标题")
        prompt_dict = {}
    
    # 如果提供了节点标题，先构建节点ID到标题的映射
    title_to_id = {}
    if target_node_titles and isinstance(prompt_dict, dict):
        for node_id, node_info in prompt_dict.items():
            if isinstance(node_info, dict) and '_meta' in node_info and 'title' in node_info['_meta']:
                title = node_info['_meta']['title']
                if title in target_node_titles:
                    title_to_id[title] = node_id
    
    # 从输出中提取文本
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'text' in node_output:
            # 默认提取所有文本输出节点
            should_extract = target_node_titles is None
            
            # 如果节点 ID 在目标节点标题对应的 ID 中
            if node_id in title_to_id.values():
                should_extract = True
            # 或者从 prompt 字典中检查节点标题
            elif isinstance(prompt_dict, dict) and node_id in prompt_dict:
                node_info = prompt_dict[node_id]
                if isinstance(node_info, dict) and '_meta' in node_info:
                    node_title = node_info['_meta'].get('title')
                    if node_title in target_node_titles:
                        should_extract = True
            
            if should_extract:
                text_outputs[node_id] = node_output['text']
    
    return text_outputs

def identify_target_nodes(workflow_json):
    """识别需要监控的目标节点"""
    target_nodes = {
        "ExtractedMessage": None,
        "BitAccuracy": None,
        "PSNR": None,
        "SSIM": None
    }
    
    # 通过节点标题查找对应的节点ID
    for node_id, node_data in workflow_json.items():
        if '_meta' in node_data and 'title' in node_data['_meta']:
            title = node_data['_meta']['title']
            for target_name in target_nodes.keys():
                if title == target_name:
                    target_nodes[target_name] = node_id
    
    return target_nodes

def identify_nodes_by_type(workflow_json, class_type_keywords):
    """按节点类型识别节点"""
    found_nodes = {}
    
    for node_id, node_data in workflow_json.items():
        if 'class_type' in node_data:
            class_type = node_data['class_type']
            for keyword in class_type_keywords:
                if keyword in class_type:
                    node_title = node_data.get('_meta', {}).get('title', node_id)
                    found_nodes[node_id] = {
                        'title': node_title,
                        'class_type': class_type
                    }
    
    return found_nodes

def print_node_structure(workflow_json):
    """打印工作流节点结构，辅助调试"""
    print("\n===== 工作流节点结构 =====")
    nodes_by_type = {}
    
    for node_id, node_data in workflow_json.items():
        if 'class_type' in node_data:
            class_type = node_data['class_type']
            if class_type not in nodes_by_type:
                nodes_by_type[class_type] = []
            
            node_title = node_data.get('_meta', {}).get('title', f'节点 {node_id}')
            nodes_by_type[class_type].append({
                'id': node_id,
                'title': node_title
            })
    
    for class_type, nodes in nodes_by_type.items():
        print(f"\n{class_type} ({len(nodes)}个节点):")
        for node in nodes:
            print(f"  - {node['title']} (ID: {node['id']})")

def identify_completion_nodes(workflow_json):
    """识别用于判断工作流完成的关键节点"""
    completion_nodes = {
        "ExtractedMessage": None,
        "BitAccuracy": None
    }
    
    # 通过节点标题查找对应的节点ID
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict) and '_meta' in node_data and 'title' in node_data['_meta']:
            title = node_data['_meta']['title']
            if title in completion_nodes:
                completion_nodes[title] = node_id
            # 额外检查与标题部分匹配的情况
            elif any(title_key in title for title_key in completion_nodes.keys()):
                for title_key in completion_nodes.keys():
                    if title_key in title:
                        completion_nodes[title_key] = node_id
    
    return completion_nodes

def get_workflow_hash(workflow_json, custom_settings=None):
    """为工作流和自定义设置生成唯一哈希值"""
    # 创建一个可序列化的工作流副本
    workflow_copy = workflow_json.copy()
    
    # 应用自定义设置到副本
    if custom_settings:
        for node_id, settings in custom_settings.items():
            if node_id in workflow_copy:
                for key, value in settings.items():
                    if 'inputs' in workflow_copy[node_id] and key in workflow_copy[node_id]['inputs']:
                        workflow_copy[node_id]['inputs'][key] = value
    
    # 序列化工作流并生成哈希
    serialized = pickle.dumps(workflow_copy)
    return hashlib.sha256(serialized).hexdigest()

def initialize_cache():
    """初始化缓存目录，首次运行时清空"""
    cache_dir = Path("comfyui_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # 检查是否首次运行
    if not FIRST_RUN_FLAG.exists():
        # 清空缓存目录
        print("首次运行，清空缓存目录...")
        for item in cache_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # 创建标志文件
        FIRST_RUN_FLAG.touch()

def check_cache(workflow_json, custom_settings=None):
    """检查是否有可用的缓存结果"""
    # 初始化缓存
    initialize_cache()
    
    # 创建缓存目录
    cache_dir = Path("comfyui_cache")
    cache_file = cache_dir / "latest_result.pkl"
    
    # 获取当前工作流的哈希值（用于比较，但不用于文件名）
    current_hash = get_workflow_hash(workflow_json, custom_settings)
    
    # 检查缓存是否存在
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # 检查缓存的哈希值是否匹配当前工作流
            if cached_data.get('hash') == current_hash:
                print(f"\n找到匹配的缓存结果")
                return cached_data.get('result')
            else:
                print(f"\n缓存存在但不匹配当前工作流设置")
        except Exception as e:
            print(f"读取缓存出错: {str(e)}")
    
    return None

def save_to_cache(result, workflow_json, custom_settings=None):
    """保存结果到缓存，仅保留最新的一个缓存文件"""
    cache_dir = Path("comfyui_cache")
    cache_file = cache_dir / "latest_result.pkl"
    
    # 获取工作流哈希值
    workflow_hash = get_workflow_hash(workflow_json, custom_settings)
    
    # 准备缓存数据，包含哈希值和结果
    cache_data = {
        'hash': workflow_hash,
        'timestamp': datetime.now().isoformat(),
        'result': result
    }
    
    try:
        # 保存为唯一的缓存文件
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"结果已缓存为最新结果")
    except Exception as e:
        print(f"缓存结果时出错: {str(e)}")

def monitor_execution_progress(ws, prompt_id, workflow_json):
    """监控工作流执行进度，添加超时检测"""
    print("\n===== 开始监控工作流执行 =====")
    
    # 创建总进度条
    progress_bar = tqdm(total=100, desc="总体进度", position=0)
    
    # 识别关键完成节点（ExtractedMessage 和 BitAccuracy）
    completion_nodes = identify_completion_nodes(workflow_json)
    completion_node_ids = {node_id for node_id in completion_nodes.values() if node_id is not None}
    
    if completion_node_ids:
        print(f"检测到完成标志节点: {', '.join(completion_nodes.keys())}")
    else:
        print("警告: 未找到完成标志节点，将等待所有节点执行完成")
    
    # 其他节点状态跟踪
    executing_nodes = set()
    completed_nodes = set()
    start_time = time.time()
    last_message_time = time.time()
    timeout_seconds = 120  # 设置超时时间为0秒
    
    try:
        while True:
            # 设置接收超时，以便定期检查整体超时
            ws.settimeout(2)
            
            try:
                out = ws.recv()
                last_message_time = time.time()  # 更新最后消息时间
                
                if isinstance(out, str):
                    message = json.loads(out)
                    
                    # 处理进度消息
                    if message['type'] == 'progress':
                        data = message['data']
                        progress = data['value']
                        max_progress = data['max']
                        
                        # 计算百分比并更新进度条
                        progress_pct = min(int(progress / max_progress * 100), 100)
                        progress_bar.update(progress_pct - progress_bar.n)
                        
                        # 显示当前正在处理的节点
                        if 'node' in data and data['node'] is not None:
                            node_id = data['node']
                            node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'节点 {node_id}')
                            tqdm.write(f"处理中: {node_title} ({progress}/{max_progress})")
                    
                    # 处理节点执行状态
                    elif message['type'] == 'executing':
                        data = message['data']
                        node_id = data['node']
                        
                        # 工作流完成信号
                        if node_id is None and data['prompt_id'] == prompt_id:
                            if 'outputs' in data and len(data['outputs']) > 0:
                                progress_bar.update(100 - progress_bar.n)  # 确保进度条达到100%
                                tqdm.write("\n工作流执行完成! (服务器信号)")
                                elapsed_time = time.time() - start_time
                                tqdm.write(f"总执行时间: {elapsed_time:.2f}秒")
                                return True
                        
                        # 节点开始执行
                        elif node_id is not None:
                            executing_nodes.add(node_id)
                            node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'节点 {node_id}')
                            tqdm.write(f"开始执行: {node_title}")
                    
                    # 处理节点完成事件
                    elif message['type'] == 'executed':
                        data = message['data']
                        node_id = data['node']
                        
                        if node_id is not None:
                            if node_id in executing_nodes:
                                executing_nodes.remove(node_id)
                                completed_nodes.add(node_id)
                                node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'节点 {node_id}')
                                
                                # 特别标记关键完成节点
                                is_completion_node = node_id in completion_node_ids
                                node_status = "完成执行" + (" (关键节点)" if is_completion_node else "")
                                tqdm.write(f"{node_status}: {node_title}")
                                
                                # 检查关键完成节点是否都已执行完成
                                if completion_node_ids and all(node_id in completed_nodes for node_id in completion_node_ids):
                                    progress_bar.update(100 - progress_bar.n)  # 确保进度条达到100%
                                    elapsed_time = time.time() - start_time
                                    tqdm.write(f"\n关键节点已全部完成! 流程结束 (用时: {elapsed_time:.2f}秒)")
                                    return True
                else:
                    # 二进制数据（预览）
                    continue
                    
            except websocket.WebSocketTimeoutException:
                # 检查是否超时
                current_time = time.time()
                if current_time - last_message_time > timeout_seconds:
                    tqdm.write(f"\n超过 {timeout_seconds} 秒无消息，假定工作流已完成")
                    # 确保进度条达到100%
                    progress_bar.update(100 - progress_bar.n)
                    return True
                continue  # 继续等待消息
                
    except websocket.WebSocketConnectionClosedException:
        tqdm.write("\n连接已关闭")
    except Exception as e:
        tqdm.write(f"\n监控进度出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    progress_bar.close()
    return False

def extract_images_with_progress(history, prompt_id):
    """提取图像并显示进度"""
    print("\n===== 提取生成的图像 =====")
    images = {}
    
    # 获取实际的 prompt 字典
    if isinstance(history['prompt'], list) and len(history['prompt']) >= 3:
        prompt_dict = history['prompt'][2]  # 第三个元素是节点字典
    else:
        print("警告: 历史记录格式不正确，无法正确解析节点标题")
        prompt_dict = {}
    
    # 统计需要提取的图像总数
    total_images = 0
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            total_images += len(node_output['images'])
    
    print(f"发现 {total_images} 张图像需要提取")
    
    # 使用总进度条
    if total_images > 0:
        overall_progress = tqdm(total=total_images, desc="总图像提取进度")
        
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                # 获取节点标题
                if isinstance(prompt_dict, dict) and node_id in prompt_dict:
                    node_info = prompt_dict[node_id]
                    node_title = node_info.get('_meta', {}).get('title', f'节点 {node_id}')
                else:
                    node_title = f'节点 {node_id}'
                    
                print(f"\n提取 {node_title} 的图像:")
                
                images_output = []
                for i, image in enumerate(node_output['images']):
                    # 更新提取信息
                    print(f"\r  - 提取图像 {i+1}/{len(node_output['images'])}: {image['filename']}", end="")
                    
                    # 获取图像
                    try:
                        image_data = get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                        overall_progress.update(1)
                    except Exception as e:
                        print(f"\n  - 提取失败: {str(e)}")
                
                print()  # 换行
                images[node_id] = images_output
        
        overall_progress.close()
    
    return images

def run_workflow_with_monitoring(workflow_file, custom_settings=None):
    """运行工作流并监控进度，添加缓存检查"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results_{timestamp}"
    
    try:
        # 加载工作流
        with open(workflow_file, 'r') as f:
            workflow_json = json.load(f)
        
        print(f"加载工作流: {workflow_file}")
        
        # 检查缓存
        cached_result = check_cache(workflow_json, custom_settings)
        if cached_result:
            print("使用缓存结果，跳过工作流执行")
            return cached_result
        
        # 应用自定义设置
        if custom_settings:
            for node_id, settings in custom_settings.items():
                if node_id in workflow_json:
                    for key, value in settings.items():
                        if 'inputs' in workflow_json[node_id] and key in workflow_json[node_id]['inputs']:
                            workflow_json[node_id]['inputs'][key] = value
                            print(f"应用设置: {node_id}.{key} = {value}")
        
        # 连接WebSocket
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}&token={TOKEN}")
        print(f"已连接到 ComfyUI 服务器: {server_address}")
        
        # 提交工作流
        print("提交工作流...")
        prompt_id = queue_prompt(workflow_json)['prompt_id']
        print(f"工作流已提交，提示ID: {prompt_id}")
        
        # 监控执行进度
        workflow_completed = monitor_execution_progress(ws, prompt_id, workflow_json)
        
        # 获取执行历史
        history = get_history(prompt_id)[prompt_id]
        
        # 识别并提取关键文本输出
        target_titles = ["ExtractedMessage", "BitAccuracy", "PSNR", "SSIM", "MSE"]
        text_outputs = extract_text_outputs(history, target_titles)
        
        # 提取生成的图像
        images = extract_images_with_progress(history, prompt_id)
        # print(type(images), images.keys())
        
        # images.update('Image-Different',diff_img)
        # print(type(images), images.keys())

        # 显示结果摘要
        display_results_summary(text_outputs, images, workflow_json)
        
        # 关闭连接
        ws.close()
        print("已关闭与服务器的连接")
        
        # 创建结果对象
        result = {
            "prompt_id": prompt_id,
            "text_outputs": text_outputs,
            "images": images,
            "completed": workflow_completed
        }
        
        # 保存到缓存
        save_to_cache(result, workflow_json, custom_settings)
        
        return result
        
    except Exception as e:
        print(f"执行工作流时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def display_results_summary(text_outputs, images, workflow_json):
    """显示结果摘要"""
    print("\n===== 结果摘要 =====")
    
    # 构建节点ID到标题的映射
    node_titles = {}
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict) and '_meta' in node_data and 'title' in node_data['_meta']:
            node_titles[node_id] = node_data['_meta']['title']
    
    # 显示水印信息
    print("\n水印信息:")
    watermark_info_found = False
    for node_id, text in text_outputs.items():
        node_title = node_titles.get(node_id, f'节点 {node_id}')
        # 检查标题是否包含关键字
        if "ExtractedMessage" in node_title or "BitAccuracy" in node_title:
            print(f"  - {node_title}: {text}")
            watermark_info_found = True
    
    if not watermark_info_found:
        print("  未找到水印信息")
    
    # 显示图像质量指标
    print("\n图像质量指标:")
    metrics_found = False
    for node_id, text in text_outputs.items():
        node_title = node_titles.get(node_id, f'节点 {node_id}')
        if "PSNR" in node_title or "SSIM" in node_title or "MSE" in node_title:
            print(f"  - {node_title}: {text}")
            metrics_found = True
    
    if not metrics_found:
        print("  未找到图像质量指标")
    
    # 显示图像输出统计
    print(f"\n生成的图像: {sum(len(imgs) for imgs in images.values())} 张")
    for node_id, node_images in images.items():
        node_title = node_titles.get(node_id, f'节点 {node_id}')
        print(f"  - {node_title} (节点 {node_id}): {len(node_images)} 张图像")

if __name__ == "__main__":
    
    
    # 自定义设置
    custom_settings = {
        "PicPrompts": {
            "text": "A young woman in her 20s stands at times square wearing a blue and yellow sweater at night."
        },
        "PicWidth": {
            "value": 1024
        },
        "PicHeight": {
            "value": 1024
        },
        "PicSeed": {
            "seed": 179
        },
        "First_stpes": {
            "steps": 12
        },
        "secondSampler": {
            "steps": 3,
            "denoise": 0.51
        },
        "Inversion_stpes": {
            "steps": 3
        },
        "WatermarkMessage": {
            "author_name": "lthero",
            "author_id": "02025LTHERO",
            "model_name": "flux",
            "Key": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
            "Nonce": "05072fd1c2265f6f2e2a4080a2bfbdd8"
        }
    }
    
    # 运行工作流并监控进度
    result = run_workflow_with_monitoring(aio_workflow_file, custom_settings)
    
    if result:
        print(f"\n工作流执行完成，提示ID: {result['prompt_id']}")
        print(f"生成了 {sum(len(imgs) for imgs in result['images'].values())} 张图像")
    else:
        print("\n工作流执行失败")
    
