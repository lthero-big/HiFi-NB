import gradio as gr
import sys
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
import argparse
from comfyui_api import (
    initialize_cache, get_workflow_hash, check_cache, save_to_cache,
    queue_prompt, get_image, get_history, extract_text_outputs,
    extract_images_with_progress, monitor_execution_progress,
    identify_completion_nodes, display_results_summary,
    server_address, client_id, TOKEN, websocket,generate_difference
)

# 工作流文件路径
DEFAULT_WORKFLOW = "workflow_api_jsons/Demo-[HiFi-NB]Flux.json"
import json

def load_workflow(workflow_file):
    with open(workflow_file, 'r') as f:
        return json.load(f)

def process_images(images):
    """处理图像数据，转换为PIL Image对象"""
    processed_images = {}
    for node_id, node_images in images.items():
        processed_images[node_id] = []
        for img_data in node_images:
            try:
                img = Image.open(io.BytesIO(img_data))
                processed_images[node_id].append(img)
            except Exception as e:
                print(f"处理图像出错: {str(e)}")
    return processed_images

def handle_seed(seed, seed_behavior):
    """根据种子行为处理种子值"""
    if seed_behavior == "递增":
        return seed + 1
    elif seed_behavior == "递减":
        return max(0, seed - 1)
    elif seed_behavior == "随机":
        return random.randint(0, 2**30)
    else:
        return seed


def run_comfyui_workflow(
    pic_prompt, width, height, seed, seed_behavior,
    first_steps, second_steps, second_denoise, inversion_steps,
    author_name, author_id, model_name,Key,Nonce, diff_scale,use_color
):
    """运行ComfyUI工作流并返回结果"""
    # 处理种子
    seed = int(seed)
    
    # 构建自定义设置
    custom_settings = {
        "PicPrompts": {
            "text": pic_prompt
        },
        "PicWidth": {
            "value": int(width)
        },
        "PicHeight": {
            "value": int(height)
        },
        "PicSeed": {
            "seed": seed
        }
    }
    
    # 添加可选参数
    if first_steps:
        custom_settings["First_stpes"] = {"steps": int(first_steps)}
    
    if second_steps:
        custom_settings["secondSampler"] = {"steps": int(second_steps)}
        if second_denoise:
            custom_settings["secondSampler"]["denoise"] = float(second_denoise)
    
    if inversion_steps:
        custom_settings["Inversion_stpes"] = {"steps": int(inversion_steps)}
    
    # 添加水印信息参数
    custom_settings["WatermarkMessage"] = {
        "author_name": author_name,
        "author_id": author_id,
        "model_name": model_name,
        "Key": Key,
        "Nonce": Nonce
    }
    
    # 加载工作流
    workflow_json = load_workflow(DEFAULT_WORKFLOW)
    
    # 检查缓存
    cached_result = check_cache(workflow_json, custom_settings)
    if cached_result:
        print("使用缓存结果")
        result = cached_result
    else:
        print("运行新工作流")
        # 连接WebSocket
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}&token={TOKEN}")
        
        # 应用自定义设置
        for node_id, settings in custom_settings.items():
            if node_id in workflow_json:
                for key, value in settings.items():
                    if 'inputs' in workflow_json[node_id] and key in workflow_json[node_id]['inputs']:
                        workflow_json[node_id]['inputs'][key] = value
        
        # 提交工作流
        prompt_id = queue_prompt(workflow_json)['prompt_id']
        
        # 监控执行进度
        workflow_completed = monitor_execution_progress(ws, prompt_id, workflow_json)
        
        # 获取执行历史，添加重试和错误处理
        max_retries = 3
        history = None
        
        for attempt in range(max_retries):
            try:
                history_data = get_history(prompt_id)
                if prompt_id in history_data:
                    history = history_data[prompt_id]
                    break
                else:
                    print(f"尝试 {attempt+1}/{max_retries}: 历史记录中未找到 prompt_id {prompt_id}，等待1秒后重试...")
                    time.sleep(1)
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_retries}: 获取历史记录出错: {str(e)}，等待1秒后重试...")
                time.sleep(1)
        
        if history is None:
            print(f"无法获取工作流历史记录，请检查 ComfyUI 服务器状态")
            return None, None, "获取历史记录失败", "请检查服务器连接", seed
        
        # 提取文本输出
        target_titles = ["ExtractedMessage", "BitAccuracy", "PSNR", "SSIM", "MSE"]
        text_outputs = extract_text_outputs(history, target_titles)
        
        # 提取图像
        images = extract_images_with_progress(history, prompt_id)
        
        # 创建结果对象
        result = {
            "prompt_id": prompt_id,
            "text_outputs": text_outputs,
            "images": images,
            "completed": workflow_completed
        }
        
        # 保存到缓存
        save_to_cache(result, workflow_json, custom_settings)
        
        # 关闭连接
        ws.close()
    
    # 处理图像
    processed_images = process_images(result["images"])
    
    # 如果处理图像为空，返回错误信息
    if not processed_images or all(len(imgs) == 0 for imgs in processed_images.values()):
        return None, None, "未生成图像", "请检查工作流设置或ComfyUI服务器状态", seed

    # 找到水印图和无水印图
    watermarked_img = None
    unwatermarked_img = None
    diff_img=None
    
    # 查找包含特定名称的节点
    for node_id, images in processed_images.items():
        if images:
            # 取第一张图片作为输出
            if "水印图" in str(node_id) or "Watermarked" in str(node_id):
                watermarked_img = images[0]
            elif "原图" in str(node_id) or "Clean" in str(node_id) or len(processed_images) == 1:
                unwatermarked_img = images[0]
    
    # 如果只有一个节点的图像，则可能需要查看前缀/后缀来确定哪个是水印图
    if unwatermarked_img is None and watermarked_img is None and processed_images:
        # 默认使用第一个节点的第一张图片
        first_node = list(processed_images.keys())[0]
        if processed_images[first_node]:
            unwatermarked_img = processed_images[first_node][0]
    

    diff_img = generate_difference(watermarked_img, unwatermarked_img, scale_factor=diff_scale,use_color=use_color)

    # 格式化水印信息
    watermark_info = ""
    quality_metrics = ""
    
    for node_id, text in result["text_outputs"].items():
        if "ExtractedMessage" in node_id :
            
            text2dict=json.loads(text[0])
            text2js = json.dumps(text2dict, sort_keys=False, indent=4, separators=(',', ':'))
            watermark_info += f"Length of Message: {len(text[0])*8} bit | {len(text[0])} Byte \n"
            watermark_info += f"{node_id}: {text2js}\n"
            # watermark_info +=text2js
            
        elif "PSNR" in node_id or "SSIM" in node_id or "MSE" in node_id or "BitAccuracy" in node_id:
            quality_metrics += f"{node_id}: {text}\n"
    
    # 更新种子
    next_seed = handle_seed(seed, seed_behavior)
    
    return unwatermarked_img, watermarked_img, diff_img, watermark_info, quality_metrics, next_seed, unwatermarked_img, watermarked_img
    
with gr.Blocks(title="Demo-水印处理") as demo:
    gr.Markdown("# Demo-水印图像生成系统")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## 输入参数")
            
            # 提示文本
            prompt = gr.Textbox(
                label="图像描述", 
                value="electric woman, cute - fine - face, pretty face, oil slick hair, realistic shaded perfect face, extremely fine details, realistic shaded lighting, dynamic background, artgerm, 8 k ultra realistic, highly detailed, art by sylvain sarrailh, alena aenami, jeremy lipkin, michael garmash, ando tadao, kan liu",
                lines=4
            )
            
            with gr.Row():
                # 图像尺寸
                width = gr.Slider(label="图片宽度", minimum=384, maximum=2048, value=1024, step=64)
                height = gr.Slider(label="图片高度", minimum=384, maximum=2048, value=1024, step=64)
            
            
            with gr.Row():
                # 种子设置
                seed = gr.Number(label="种子(相同种子会生成相同图像)", value=178, precision=0)
                seed_behavior = gr.Dropdown(
                    label="每次生成后种子变化方式", 
                    choices=["保持不变", "自动递增", "自动递减", "随机"], 
                    value="随机"
                )
            
            # 采样参数
            with gr.Row():
                first_steps = gr.Number(label="去噪步数(正常业务流程,取5~20步)", value=12, precision=0,minimum=1,maximum=50)
                inversion_steps = gr.Number(label="[水印提取]反转步数", value=3, precision=0,minimum=1,maximum=50)
            
            with gr.Row():
                second_steps = gr.Number(label="[水印嵌入]去噪步数-用于优化水印后的图像质量", value=3, precision=0,minimum=1,maximum=50)
                second_denoise = gr.Slider(
                    label="水印保留强度(越小水印越无法保留)", 
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.51, 
                    step=0.01
                )
            
            # 水印信息参数
            gr.Markdown("### 水印信息")
            with gr.Row():
                author_name = gr.Textbox(label="作者名称", value="lthero")
                author_id = gr.Textbox(label="作者ID", value="02025LTHERO")
                model_name = gr.Textbox(label="模型名称", value="flux")

            gr.Markdown("### 水印密钥")
            gr.Markdown("使用相同密钥才能正确提取水印信息。")
            with gr.Row():    
                Key=gr.Textbox(label="主加密密钥（建议不修改默认值）", value="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7")
                Nonce=gr.Textbox(label="初始化向量（建议不修改默认值）", value="05072fd1c2265f6f2e2a4080a2bfbdd8")
            
        with gr.Column(scale=4):
            gr.Markdown("## 生成图片")
            with gr.Row():
                # 显示无水印和有水印图像
                original_img = gr.Image(label="原始图像 (无水印)", type="pil", height=512)
                watermarked_img = gr.Image(label="水印图像", type="pil", height=512)
                # 添加隐藏的复制组件，仅用于差值图生成的输入
                original_img_for_diff = gr.Image(visible=False)
                watermarked_img_for_diff = gr.Image(visible=False)
            # 生成按钮
            generate_btn = gr.Button("生成图像", variant="primary")
            # 水印信息和质量指标显示（移到按钮下方）
            gr.Markdown("## 水印信息与图像质量评价")
            with gr.Row():
                watermark_info = gr.Code(label="水印信息", language="json")
                quality_metrics = gr.Code(label="图像质量指标", language="json")

        with gr.Column(scale=2):
            gr.Markdown("## 差值图像")
            diff_img = gr.Image(label="差值图像", type="pil", height=512)
            diff_scale=gr.Slider(label="差值图的差值放大倍数", minimum=1, maximum=100, value=30, step=1)
            use_color=gr.Checkbox(label="是否使用颜色",value=False)
            diff_img_bt=gr.Button("生成差值图",variant="primary")
            
    # 生成差值图
    diff_img_bt.click(
        fn=generate_difference,
        inputs=[watermarked_img_for_diff, original_img_for_diff, diff_scale, use_color],
        outputs=[
            diff_img
        ]
    )
    
    # 设置函数调用
    generate_btn.click(
        fn=run_comfyui_workflow,
        inputs=[
            prompt, width, height, seed, seed_behavior,
            first_steps, second_steps, second_denoise, inversion_steps,
            author_name, author_id, model_name, Key, Nonce, diff_scale, use_color
        ],
        outputs=[
            original_img, watermarked_img, diff_img, watermark_info, quality_metrics, seed, 
            original_img_for_diff, watermarked_img_for_diff 
        ]
    )
    
    
    


# 初始化缓存
initialize_cache()

# 启动界面
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Demo-水印图像生成系统')
    parser.add_argument('--listen', action='store_true', help='监听所有网络接口 (0.0.0.0)')
    parser.add_argument('--port', type=int, default=8180, help='Web服务端口号')
    parser.add_argument('--share', action='store_true', help='生成公共链接(通过Gradio)')
    parser.add_argument('--username', type=str, default="dongli911", help='登录用户名')
    parser.add_argument('--password', type=str, default="Dongli@911", help='登录密码')
    args = parser.parse_args()
    
    # 配置启动参数
    server_name = "0.0.0.0" if args.listen else None
    server_port = args.port
    
    # 处理认证信息
    auth = None
    if args.username is not None or args.password is not None:
        # 确保同时提供了用户名和密码
        if args.username is None or args.password is None:
            print("错误: 必须同时提供用户名和密码")
            sys.exit(1)
        auth = (args.username, args.password)
        print(f"已启用登录保护，用户名: {args.username}")
    
    # 用于测试时跳过验证
    auth =None
    print(f"启动Web服务 - {'监听所有网络接口' if args.listen else '仅本地访问'}, 端口: {server_port}")
    if auth:
        print("访问需要登录认证")
    
    # 启动Gradio应用
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=args.share,
        auth=auth  # 添加认证参数
    )