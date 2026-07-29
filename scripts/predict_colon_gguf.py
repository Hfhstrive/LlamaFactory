import os
import sys
import json
import re
import time
import psutil
import torch
import traceback

try:
    from llama_cpp import Llama
except ImportError as e:
    print("\n" + "="*60)
    print("[导入错误] 无法导入 llama_cpp 模块！")
    print("错误详情:")
    traceback.print_exc()
    print("="*60)
    print("\n这通常是因为以下两个原因之一：")
    print("1. 您运行该脚本的 Python 解释器（环境）与您刚刚运行 pip install 的环境不一致。")
    print("2. 缺少 CUDA 运行时动态库链接（例如 libcudart.so），请检查上面的错误栈。")
    sys.exit(1)

def main():
    # 1. 指定 GGUF 模型文件与推理参数
    # 请确保已将 LoRA 模型合并并转换/量化为 .gguf 文件
    gguf_model_path = "/media/inno/work_dirs/LLM/LlamaFactory/colon/outputs-sft-qwen3.5-4b-v4-lora-64-128-0.1-warmup-0.05-decay-0.01-batch4/gguf/model_q8_0.gguf"
    
    # 推理结果 JSON 保存路径
    output_json_path = "/media/inno/output/LLM/colon/qwen3.5-4b-v4-llamafactory-lora-64-128-0.1-warmup-0.05-decay-0.01-batch4-e2/val_q8_0.json"

    n_ctx = 4096           # 上下文窗口大小
    n_gpu_layers = -1      # GPU 卸载层数，-1 表示全部层卸载至 GPU（若 GPU 显存足够）

    # 2. 加载 GGUF 模型
    print(f"正在通过 llama_cpp 加载 GGUF 模型: {gguf_model_path}...")
    start_init_time = time.time()
    
    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=True
    )
    
    end_init_time = time.time()
    init_duration = end_init_time - start_init_time
    print(f"GGUF 模型加载完成，初始化耗时: {init_duration:.2f} 秒。")

    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        print(f"模型加载后 PyTorch 侧 GPU 显存占用: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")

    # 3. 收集输入评估数据
    inputs_to_run = {}
    asr_path = '/media/inno/LLM/肠镜/report/V3/ann'
    val_json_path = os.path.join(asr_path, 'val.json')
    if os.path.exists(val_json_path):
        print(f"正在从 {val_json_path} 读取评估数据...")
        with open(val_json_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            for key, text in val_data.items():
                inputs_to_run[key] = text
    else:
        print(f"警告：未找到评估文件 {val_json_path}")

    print(f"总计收集到待测样例数: {len(inputs_to_run)}")

    results = {}
    total_generated_tokens = 0

    # 4. 循环进行 GGUF 推理
    print("\n开始 GGUF 模型推理评估样例...")
    start_inference_time = time.time()

    for key, asr_text in inputs_to_run.items():
        question = f"提取有效信息,生成标准肠镜报告：{asr_text}"
        messages = [{"role": "user", "content": question}]
        
        sample_start_time = time.time()
        
        # 4.1 调用 llama_cpp 的 create_chat_completion API 进行生成 (temperature=0.0 表示贪婪解码)
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        response_text = completion["choices"][0]["message"]["content"] or ""
        response_len = completion.get("usage", {}).get("completion_tokens", 0)
        total_generated_tokens += response_len

        # 4.2 解析并分离 <think>...</think> 思考过程与最终输出结果
        think_content = ""
        output_content = response_text.strip()
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            output_content = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        print(f"\n--- Key: {key} ---")
        print(f"问：{question}")
        if think_content:
            print(f"思考过程：\n{think_content}")
        print(f"最终输出：\n{output_content}")
        
        # 结构化保存：同时保留思考过程、最终输出以及原始文本
        results[key] = {
            "think": think_content,
            "output": output_content,
            "raw": response_text.strip()
        }
        
        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time
        print(f"[耗时统计] 样本 {key} GGUF 推理耗时: {sample_duration:.2f} 秒 (生成 {response_len} tokens)")
        print('----------------------------------------------------------------------------------')

    end_inference_time = time.time()
    total_inference_duration = end_inference_time - start_inference_time

    # 5. 获取 CPU 使用占比与内存指标
    cpu_percent = psutil.cpu_percent(interval=0.5)
    process = psutil.Process(os.getpid())
    process_cpu_percent = process.cpu_percent(interval=None)

    # 打印运行性能报告
    print("\n================== GGUF 性能指标汇总 ==================")
    print(f"1. GGUF 模型加载耗时: {init_duration:.2f} 秒")
    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        print(f"2. GPU 显存占用量: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")
    print(f"3. 系统 CPU 占用比例: {cpu_percent:.1f}%")
    print(f"   当前进程 CPU 比例: {process_cpu_percent:.1f}%")

    num_samples = len(inputs_to_run)
    if num_samples > 0:
        avg_sample_time = total_inference_duration / num_samples
        avg_tokens_per_sec = total_generated_tokens / total_inference_duration if total_inference_duration > 0 else 0.0
        print(f"4. 总推理耗时: {total_inference_duration:.2f} 秒")
        print(f"5. 总生成 Token 数量: {total_generated_tokens}")
        print(f"6. 整体推理吞吐速度: {avg_tokens_per_sec:.2f} tokens/s")
        print(f"7. 平均单个 asr_text 的推理耗时: {avg_sample_time:.2f} 秒/样本")
    print("======================================================\n")

    # 6. 保存结果到单 JSON 文件中
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"所有 GGUF 推理结果已成功保存至: {output_json_path}")

if __name__ == "__main__":
    main()
