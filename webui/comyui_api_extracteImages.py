from tqdm import tqdm
import time
import sys
from datetime import datetime
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import hashlib
import pickle
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import csv
import re

# Configuration
server_address = "127.0.0.1:8189"
client_id = str(uuid.uuid4())
TOKEN = "$2b$12$ixMAtJWcH3bvLeJMLESgyevl0mbkITbo3l.pE0ct7thgRq4D3tTlW"
FIRST_RUN_FLAG = Path("comfyui_cache/.first_run")
dataSet = "SD2"  # "SD2" "Flux" or "Laion"
waterMethod = "GS" # "GS" or "DPRW"
workflow_file = f"workflow_api_jsons/Demo-[DPRW]Flux_extraction_FromUPLoad_and_DDIM_inversion.json"
bit_length=512
dict_bit_length_message={
    32: "lthe",
    64: "Testing.",
    128: "It's a test case",
    256: "It's a test to see if this works",
    512: "This sentence has exactly sixty-four characters but it's not...."
}
# input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/ablation/"
# input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/Exist_Image/Laion/100pics/HiFiNB_512bit"
# input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/20pics-flux-images/GS-512bit"
# input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/20pics-SD2-images/HiFi-NB_SD2_512bit"
input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/30pics-SD2-images/GS_SD2_512bit"

# input_dir = f"/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/100pics_HiFi-NB_Flux-DiffP_{bit_length}bit_noise/" 
# input_dir = f"/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/Exist_Image/Laion/100pics/100pics-Flux-Image-Laion-{bit_length}bit"
traverse_dir_flag = False
# input_dir = f"/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/100pics_HiFi-NB_Flux-DiffP_512bit_noise_temp/" 
# input_dir = "/home/dongli911/.wan/Project/AIGC/DPRW/Experiments/test_256bit_noise/"
key = "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
nonce = "05072fd1c2265f6f2e2a4080a2bfbdd8"
# original_message = "It's a test to see if this works"
original_message = dict_bit_length_message[bit_length]

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt?token={TOKEN}", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}?token={TOKEN}") as response:
        return json.loads(response.read())

def get_workflow_hash(workflow_json, custom_settings=None):
    workflow_copy = workflow_json.copy()
    if custom_settings:
        for node_id, settings in custom_settings.items():
            if node_id in workflow_copy:
                for key, value in settings.items():
                    if 'inputs' in workflow_copy[node_id] and key in workflow_copy[node_id]['inputs']:
                        workflow_copy[node_id]['inputs'][key] = value
    serialized = pickle.dumps(workflow_copy)
    return hashlib.sha256(serialized).hexdigest()

def extract_text_outputs(history, target_node_titles=None):
    text_outputs = {}
    if isinstance(history['prompt'], list) and len(history['prompt']) >= 3:
        prompt_dict = history['prompt'][2]
    else:
        prompt_dict = {}
    
    title_to_id = {}
    if target_node_titles and isinstance(prompt_dict, dict):
        for node_id, node_info in prompt_dict.items():
            if isinstance(node_info, dict) and '_meta' in node_info and 'title' in node_info['_meta']:
                title = node_info['_meta']['title']
                if title in target_node_titles:
                    title_to_id[title] = node_id
    
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'text' in node_output:
            should_extract = target_node_titles is None
            if node_id in title_to_id.values():
                should_extract = True
            elif isinstance(prompt_dict, dict) and node_id in prompt_dict:
                node_info = prompt_dict[node_id]
                if isinstance(node_info, dict) and '_meta' in node_info:
                    node_title = node_info['_meta'].get('title')
                    if node_title in target_node_titles:
                        should_extract = True
            if should_extract:
                text_outputs[node_id] = node_output['text']
    
    return text_outputs

def identify_completion_nodes(workflow_json):
    completion_nodes = {
        "ExtractedMessage": None,
        "BitAccuracy": None
    }
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict) and '_meta' in node_data and 'title' in node_data['_meta']:
            title = node_data['_meta']['title']
            if title in completion_nodes:
                completion_nodes[title] = node_id
            elif any(title_key in title for title_key in completion_nodes.keys()):
                for title_key in completion_nodes.keys():
                    if title_key in title:
                        completion_nodes[title_key] = node_id
    return completion_nodes

def initialize_cache():
    cache_dir = Path("comfyui_cache")
    cache_dir.mkdir(exist_ok=True)
    if not FIRST_RUN_FLAG.exists():
        print("First run, clearing cache directory...")
        for item in cache_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        FIRST_RUN_FLAG.touch()

def check_cache(workflow_json, custom_settings=None):
    initialize_cache()
    cache_dir = Path("comfyui_cache")
    cache_file = cache_dir / "latest_result.pkl"
    current_hash = get_workflow_hash(workflow_json, custom_settings)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if cached_data.get('hash') == current_hash:
                print("Found matching cached result")
                return cached_data.get('result')
        except Exception as e:
            print(f"Error reading cache: {str(e)}")
    return None

def save_to_cache(result, workflow_json, custom_settings=None):
    cache_dir = Path("comfyui_cache")
    cache_file = cache_dir / "latest_result.pkl"
    workflow_hash = get_workflow_hash(workflow_json, custom_settings)
    cache_data = {
        'hash': workflow_hash,
        'timestamp': datetime.now().isoformat(),
        'result': result
    }
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Result cached as latest result")
    except Exception as e:
        print(f"Error caching result: {str(e)}")

def monitor_execution_progress(ws, prompt_id, workflow_json):
    print("\n===== Monitoring Workflow Execution =====")
    progress_bar = tqdm(total=100, desc="Overall Progress", position=0)
    completion_nodes = identify_completion_nodes(workflow_json)
    completion_node_ids = {node_id for node_id in completion_nodes.values() if node_id is not None}
    
    if completion_node_ids:
        print(f"Detected completion nodes: {', '.join(completion_nodes.keys())}")
    else:
        print("Warning: No completion nodes found, waiting for all nodes to complete")
    
    executing_nodes = set()
    completed_nodes = set()
    start_time = time.time()
    last_message_time = time.time()
    timeout_seconds = 120
    
    try:
        while True:
            ws.settimeout(2)
            try:
                out = ws.recv()
                last_message_time = time.time()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'progress':
                        data = message['data']
                        progress = data['value']
                        max_progress = data['max']
                        progress_pct = min(int(progress / max_progress * 100), 100)
                        progress_bar.update(progress_pct - progress_bar.n)
                        if 'node' in data and data['node'] is not None:
                            node_id = data['node']
                            node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'Node {node_id}')
                            tqdm.write(f"Processing: {node_title} ({progress}/{max_progress})")
                    elif message['type'] == 'executing':
                        data = message['data']
                        node_id = data['node']
                        if node_id is None and data['prompt_id'] == prompt_id:
                            if 'outputs' in data and len(data['outputs']) > 0:
                                progress_bar.update(100 - progress_bar.n)
                                tqdm.write("\nWorkflow execution completed! (Server signal)")
                                elapsed_time = time.time() - start_time
                                tqdm.write(f"Total execution time: {elapsed_time:.2f} seconds")
                                return True
                        elif node_id is not None:
                            executing_nodes.add(node_id)
                            node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'Node {node_id}')
                            tqdm.write(f"Starting execution: {node_title}")
                    elif message['type'] == 'executed':
                        data = message['data']
                        node_id = data['node']
                        if node_id is not None:
                            if node_id in executing_nodes:
                                executing_nodes.remove(node_id)
                                completed_nodes.add(node_id)
                                node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'Node {node_id}')
                                is_completion_node = node_id in completion_node_ids
                                node_status = "Completed execution" + (" (key node)" if is_completion_node else "")
                                tqdm.write(f"{node_status}: {node_title}")
                                if completion_node_ids and all(node_id in completed_nodes for node_id in completion_node_ids):
                                    progress_bar.update(100 - progress_bar.n)
                                    elapsed_time = time.time() - start_time
                                    tqdm.write(f"\nKey nodes completed! Workflow finished (Time: {elapsed_time:.2f} seconds)")
                                    return True
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                current_time = time.time()
                if current_time - last_message_time > timeout_seconds:
                    tqdm.write(f"\nNo messages for {timeout_seconds} seconds, assuming workflow completed")
                    progress_bar.update(100 - progress_bar.n)
                    return True
                continue
    except websocket.WebSocketConnectionClosedException:
        tqdm.write("\nConnection closed")
    except Exception as e:
        tqdm.write(f"\nError monitoring progress: {str(e)}")
        import traceback
        traceback.print_exc()
    
    progress_bar.close()
    return False

def process_image(image_path, workflow_json):
    custom_settings = {
        "317": {
            "image": image_path
        },
        "203": {
            "string": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
        },
        "206": {
            "string": "05072fd1c2265f6f2e2a4080a2bfbdd8"
        },
        "209": {
            "string": original_message
        },
        "179": {
            "message_length": len(original_message)
        }
    }
    
    cached_result = check_cache(workflow_json, custom_settings)
    if cached_result:
        return cached_result
    
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}&token={TOKEN}")
    
    for node_id, settings in custom_settings.items():
        if node_id in workflow_json:
            for key, value in settings.items():
                if 'inputs' in workflow_json[node_id] and key in workflow_json[node_id]['inputs']:
                    workflow_json[node_id]['inputs'][key] = value
    
    prompt_id = queue_prompt(workflow_json)['prompt_id']
    workflow_completed = monitor_execution_progress(ws, prompt_id, workflow_json)
    
    history = get_history(prompt_id)[prompt_id]
    target_titles = ["ExtractedMessage", "BitAccuracy"]
    text_outputs = extract_text_outputs(history, target_titles)
    
    ws.close()
    
    result = {
        "prompt_id": prompt_id,
        "text_outputs": text_outputs,
        "completed": workflow_completed
    }
    
    save_to_cache(result, workflow_json, custom_settings)
    return result



def traverse_directories(root_dir)->dict:
    '''
    递归遍历指定文件夹下的所有子目录
    @root_dir :
    @return: dict['distortion_strength']=full_subdir_path
    
    '''
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"dir {root_dir} is not exist")
    
    paths_dict = {}

    try:
        for distortion in os.listdir(root_dir):
            distortion_path=os.path.join(root_dir,distortion)
            if os.path.isdir(distortion_path):
                for strength in os.listdir(distortion_path):
                    full_subdir_path=os.path.join(distortion_path,strength)
                    if os.path.isdir(full_subdir_path):
                        key=f"{distortion}_{strength}"
                        paths_dict[key]=full_subdir_path
    except PermissionError as e:
        print(f"权限错误 {e}")
    except Exception as e:
        print(f"发生错误 {e}")
    
    return paths_dict


def write_to_avg_csv_result(avg_csv_relative_path, avg_result):
    if not os.path.exists(avg_csv_relative_path):
        with open(avg_csv_relative_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['distortion_strength', 'avg_bit_accuracy', 'tpr_at_1pct_fpr']  
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(avg_result)
    else:
        with open(avg_csv_relative_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['distortion_strength', 'avg_bit_accuracy', 'tpr_at_1pct_fpr'] 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(avg_result)
    print(f"{avg_result['distortion_strength']} data saved to {os.path.basename(avg_csv_relative_path)}")


from scipy.special import betainc

def string_to_bits(message_str):
    """
    Convert a string to its binary representation
    
    Args:
        message_str (str): The message string to convert
        
    Returns:
        str: Binary representation as a string of '0' and '1' characters
    """
    # Convert string to bytes, then to binary representation
    message_bytes = message_str.encode('utf-8')
    binary = ''.join(format(byte, '08b') for byte in message_bytes)
    return binary

def calculate_tpr_at_fpr(original_message, extracted_messages, target_fpr=0.01):
    original_message = string_to_bits(original_message)
    # print(extracted_messages)
    extracted_messages = [string_to_bits(msg) for msg in extracted_messages]
    """
    Calculate TPR at a specified FPR (default 1%)
    
    Args:
        original_message (str): The original watermark message
        extracted_messages (list): List of extracted messages from watermarked images
        target_fpr (float): Target false positive rate (default 0.000001 for 1%)
    
    Returns:
        dict: Contains threshold, TPR, and other statistics
    """
    message_length = len(original_message)
    
    # Calculate bit accuracies - how many bits match between original and extracted
    accuracies = []
    for extracted in extracted_messages:
        if not extracted or len(extracted) != message_length:
            continue
            
        # Calculate bit-by-bit accuracy
        correct_bits = sum(o == e for o, e in zip(original_message, extracted))
        accuracy = correct_bits / message_length
        accuracies.append(accuracy)
    
    if not accuracies:
        return {"error": "No valid extracted messages for comparison"}
    
    # Calculate threshold based on betainc (as in the reference implementation)
    # Find threshold where FPR <= target_fpr
    threshold = None
    for i in range(message_length + 1):
        fpr = betainc(i+1, message_length-i, 0.5)
        if fpr <= target_fpr:
            threshold = i / message_length
            break
    
    if threshold is None:
        return {"error": "Could not determine threshold for target FPR"}
    
    # Calculate TPR - portion of watermarked images detected at this threshold
    detected = sum(1 for acc in accuracies if acc >= threshold)
    tpr = detected / len(accuracies)
    print(f"TPR at {target_fpr*100}% FPR: {tpr:.4f} (Threshold: {threshold:.4f}) (detected: {detected})")
    
    return {
        "threshold": threshold,
        "tpr_at_1pct_fpr": tpr,
        "detected_count": detected,
        "total_count": len(accuracies),
        "average_accuracy": np.mean(accuracies),
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies)
    }


def main():
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return
    
    with open(workflow_file, 'r') as f:
        workflow_json = json.load(f)
    
    image_extensions = (".jpg", ".jpeg", ".png")
    
    if traverse_dir_flag:
        subdirs_dict=traverse_directories(input_dir)
    else:
        subdirs_dict = {os.path.basename(input_dir): input_dir}

    print(subdirs_dict)
    if subdirs_dict is None:
        print(f"Directory {input_dir} does not contain any subdirectories.")
        return
    save_name = f"{dataSet}-{waterMethod}-{bit_length}bit"
    csv_saved_path=f"extracted_results_csv_{save_name}"
    avg_csv_name=f'results_avg_distortion_strength.csv'
    avg_csv_relative_path=os.path.join(csv_saved_path,avg_csv_name)
    
    tpr_results = {}

    for each_distortion_strength,subdir_full_path in subdirs_dict.items():
        each_distortion_strength_results = []
        print(f"\n Processing the direcotry of {each_distortion_strength}")
        image_files = [f for f in os.listdir(subdir_full_path) if f.lower().endswith(image_extensions)]

        for filename in tqdm(image_files, desc="Processing Images"):
            image_path = os.path.join(subdir_full_path, filename)
            print(f"\nProcessing the image of{filename}...")
            
            result = process_image(image_path, workflow_json.copy())
            if result and result["completed"]:
                bit_accuracy = None
                extracted_message = None
                extracted_message_binary = None
                original_message_binary = None
                for node_id, text in result["text_outputs"].items():
                    node_title = workflow_json.get(node_id, {}).get('_meta', {}).get('title', f'Node {node_id}')
                    if node_title == "BitAccuracy":
                        bit_accuracy = text
                    elif node_title == "ExtractedMessage":
                        extracted_message = text
                    elif node_title == "ExtractedMessageBinary":
                        extracted_message_binary = text
                    elif node_title == "OriginalMessageBinary":
                        original_message_binary = text
                
                # to string
                bit_accuracy=bit_accuracy[0]
                print(bit_accuracy)
                print(extracted_message[0])
                print(extracted_message_binary)
                print(original_message_binary)
                each_distortion_strength_results.append({
                    "image": filename,
                    "bit_accuracy": float(bit_accuracy),
                    "extracted_message": extracted_message[0],
                    "ExtractedMessageBinary": extracted_message_binary,
                    "OriginalMessageBinary":original_message_binary
                })
        
        
        print("\nResults:")
        for result in each_distortion_strength_results:
            print(f"{result['image']}")
            print(float(bit_accuracy))
            print(f"{result['extracted_message']}")
            print(f"{result['ExtractedMessageBinary']}")
            print(f"{result['OriginalMessageBinary']}")
            print("-" * 50)


        if not os.path.exists(csv_saved_path):
            os.mkdir(csv_saved_path)
        each_csv_name=f'results_{each_distortion_strength}.csv'
        each_csv_relative_path=os.path.join(csv_saved_path,each_csv_name)
        with open(each_csv_relative_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image', 'bit_accuracy', 'extracted_message',"ExtractedMessageBinary","OriginalMessageBinary"]  
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in each_distortion_strength_results:
                writer.writerow(result)
        print(f"数据已成功保存到 {each_csv_name} 文件中。")


        extracted_messages = [result["extracted_message"] for result in each_distortion_strength_results]
        # Calculate TPR@1%FPR
        tpr_stats = calculate_tpr_at_fpr(original_message, extracted_messages)
        tpr_results[each_distortion_strength] = tpr_stats

        each_distortion_strength_bit_accuracies=[float(item["bit_accuracy"]) for item in each_distortion_strength_results]
        if each_distortion_strength_bit_accuracies:
            avg_bit_accuracy=np.mean(each_distortion_strength_bit_accuracies)
        else:
            avg_bit_accuracy=0
        avg_result = {
            "distortion_strength": each_distortion_strength,
            "avg_bit_accuracy": avg_bit_accuracy,
            "tpr_at_1pct_fpr": tpr_stats.get("tpr_at_1pct_fpr", 0)
        }
        try:
            write_to_avg_csv_result(avg_csv_relative_path,avg_result)
        except Exception as e:
            print("Error on write to avg csv result")
            pass

        # Save overall TPR results
        tpr_csv_path = os.path.join(csv_saved_path, 'tpr_results.csv')
        with open(tpr_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['distortion_strength', 'threshold', 'tpr_at_1pct_fpr', 
                        'detected_count', 'total_count', 'average_accuracy',"min_accuracy","max_accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ds, stats in tpr_results.items():
                if "error" not in stats:
                    writer.writerow({
                        'distortion_strength': ds,
                        'threshold': stats['threshold'],
                        'tpr_at_1pct_fpr': stats['tpr_at_1pct_fpr'],
                        'detected_count': stats['detected_count'],
                        'total_count': stats['total_count'],
                        'average_accuracy': stats['average_accuracy'],
                        "min_accuracy": stats['min_accuracy'],
                        "max_accuracy": stats['max_accuracy'],
                    })
        
        print(f"TPR results saved to {tpr_csv_path}")

    

if __name__ == "__main__":
    main()