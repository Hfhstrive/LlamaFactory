import os
import json
import time
import psutil
import torch
from llamafactory.chat import ChatModel

def main():
    # 1. 指定推理参数 (与微调 YAML 配置相对应)
    args = {
        "model_name_or_path": "/home/inno/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B/",
        "adapter_name_or_path": "/media/inno/work_dirs/LLM/LlamaFactory/colon/outputs-sft-qwen3.5-4b-v1/checkpoint-77/",  # LLaMA-Factory 的微调输出路径
        "template": "qwen3_5_nothink",
        "finetuning_type": "lora",
        "trust_remote_code": True,
        # 可以添加其他推理生成参数，如:
        # "temperature": 0.0,
        # "top_p": 0.9,
    }

    # 2. 加载模型与 LoRA 适配器
    print("正在通过 LLaMA-Factory 的 ChatModel 加载模型和适配器...")
    start_init_time = time.time()
    chat_model = ChatModel(args)
    end_init_time = time.time()
    init_duration = end_init_time - start_init_time
    print(f"模型加载完成，初始化耗时: {init_duration:.2f} 秒。")

    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        print(f"模型加载后 GPU 显存占用: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")

    # 3. 收集输入数据
    inputs_to_run = {}

    # 3.1 从 val.json 读取输入
    asr_path = '/media/inno/LLM/肠镜/report/V2/ann'
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

    # 结果输出路径 (输出到 LLaMA-Factory 对应路径中)
    output_json_path = '/media/inno/output/LLM/colon/qwen3.5-4b-v1-llamafactory/val.json'
    results = {}
    total_generated_tokens = 0

    # 4. 循环进行推理
    print("\n开始推理评估样例...")
    start_inference_time = time.time()

    for key, asr_text in inputs_to_run.items():
        question = f"提取有效信息,生成标准肠镜报告：{asr_text}"
        messages = [{"role": "user", "content": question}]
        
        sample_start_time = time.time()
        
        # 4.1 调用 LLaMA-Factory API 进行生成 (do_sample=False 表示贪婪解码，对应原参数中的 do_sample=False)
        responses = chat_model.chat(messages, do_sample=False, max_new_tokens=512)
        response_text = responses[0].response_text
        response_len = responses[0].response_length
        total_generated_tokens += response_len

        print(f"\n--- Key: {key} ---")
        print(f"问：{question}")
        print(f"答：{response_text}")
        
        results[key] = response_text.strip()
        
        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time
        print(f"[耗时统计] 样本 {key} 推理耗时: {sample_duration:.2f} 秒")
        print('----------------------------------------------------------------------------------')

    end_inference_time = time.time()
    total_inference_duration = end_inference_time - start_inference_time

    # 5. 获取 CPU 使用占比和内存指标
    cpu_percent = psutil.cpu_percent(interval=0.5)
    process = psutil.Process(os.getpid())
    process_cpu_percent = process.cpu_percent(interval=None)

    # 打印运行性能报告
    print("\n================== 性能指标汇总 ==================")
    print(f"1. 模型加载/初始化耗时: {init_duration:.2f} 秒")
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
    print("=================================================\n")

    # 6. 保存结果到单 JSON 文件中
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"所有推理结果已成功保存至: {output_json_path}")

if __name__ == "__main__":
    main()
