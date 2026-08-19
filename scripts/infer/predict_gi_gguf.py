import os
import sys
import json
import re
import time
import psutil
import torch
import traceback
from collections import Counter

try:
    from llama_cpp import Llama
except ImportError as e:
    print("\n" + "=" * 60)
    print("[导入错误] 无法导入 llama_cpp 模块！")
    print("错误详情:")
    traceback.print_exc()
    print("=" * 60)
    print("\n这通常是因为以下两个原因之一：")
    print("1. 您运行该脚本的 Python 解释器（环境）与您刚刚运行 pip install 的环境不一致。")
    print("2. 缺少 CUDA 运行时动态库链接（例如 libcudart.so），请检查上面的错误栈。")
    sys.exit(1)


def parse_think_and_output(text: str):
    """
    解析文本中 <think>...</think> 思考过程与最终输出内容（鲁棒容错解析）
    """
    text = (text or "").strip()
    if "</think>" in text:
        parts = text.split("</think>", 1)
        think_part = parts[0].strip()
        output_part = parts[1].strip()
        if think_part.startswith("<think>"):
            think_part = think_part[len("<think>"):].strip()
        return think_part, output_part
    else:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            output_content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return think_content, output_content
        return "", text


def extract_procedure_and_diagnosis(output_str: str):
    """
    自适应解析内镜报告 JSON 字符串，分离出 '检查过程/镜检所见' 和 '检查结果/诊断结论'
    """
    output_str = (output_str or "").strip()
    procedure = ""
    diagnosis = ""

    try:
        data = json.loads(output_str)
        if isinstance(data, dict):
            # 兼容胃镜 (检查过程/检查结果) 与 肠镜 (镜检所见/诊断结论)
            proc_val = data.get("检查过程") if "检查过程" in data else data.get("镜检所见")
            if isinstance(proc_val, (dict, list)):
                procedure = json.dumps(proc_val, ensure_ascii=False, indent=2)
            elif proc_val is not None:
                procedure = str(proc_val)

            diag_val = data.get("检查结果") if "检查结果" in data else data.get("诊断结论")
            if isinstance(diag_val, (dict, list)):
                diagnosis = json.dumps(diag_val, ensure_ascii=False, indent=2)
            elif diag_val is not None:
                diagnosis = str(diag_val)
    except Exception:
        pass

    if not procedure and not diagnosis:
        procedure = output_str

    return procedure, diagnosis


def compute_cer(gt_text: str, pred_text: str):
    """
    计算字符错误率 CER (Character Error Rate) 与 字符准确率 (Accuracy)
    """
    gt_chars = list(gt_text or "")
    pred_chars = list(pred_text or "")
    n = len(gt_chars)
    m = len(pred_chars)

    if n == 0:
        return 0.0, 1.0 if m == 0 else 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_chars[i - 1] == pred_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_dist = dp[n][m]
    cer = edit_dist / n
    acc = max(0.0, 1.0 - cer)
    return cer, acc


def compute_char_f1(gt_text: str, pred_text: str):
    """
    计算字符级 Precision, Recall, F1-Score
    """
    gt_text = gt_text or ""
    pred_text = pred_text or ""

    gt_counts = Counter(gt_text)
    pred_counts = Counter(pred_text)

    overlap = sum((gt_counts & pred_counts).values())
    total_gt = len(gt_text)
    total_pred = len(pred_text)

    precision = overlap / total_pred if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
    recall = overlap / total_gt if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def load_val_samples(val_file_path: str):
    """
    自适应读取 JSONL 或 JSON 格式的验证集文件，兼容 conversations 与 messages 结构
    提取 system、human(user) 的输入以及 gpt(assistant) 的真实标注 (GT)
    """
    samples = []
    if not os.path.exists(val_file_path):
        alt_path = val_file_path.replace('.json', '.jsonl') if val_file_path.endswith('.json') else val_file_path.replace('.jsonl', '.json')
        if os.path.exists(alt_path):
            val_file_path = alt_path
        else:
            raise FileNotFoundError(f"未找到验证集数据文件: {val_file_path}")

    print(f"正在读取验证集文件: {val_file_path}...")
    if val_file_path.endswith('.jsonl'):
        with open(val_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(val_file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

    parsed_data = []
    default_sys = "你是一个严谨的消化内镜专家，请精准提取口语中的病变部位、特征描述与诊断结论，严禁漏诊与误诊。"

    for item in samples:
        sys_prompt = default_sys
        user_input = ""
        gt_text = ""

        if 'conversations' in item:
            for msg in item['conversations']:
                if msg.get('from') in ['system']:
                    sys_prompt = msg.get('value', sys_prompt)
                elif msg.get('from') in ['human', 'user']:
                    user_input = msg.get('value', '')
                elif msg.get('from') in ['gpt', 'assistant']:
                    gt_text = msg.get('value', '')
        elif 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') in ['system']:
                    sys_prompt = msg.get('content', sys_prompt)
                elif msg.get('role') in ['human', 'user']:
                    user_input = msg.get('content', '')
                elif msg.get('role') in ['gpt', 'assistant']:
                    gt_text = msg.get('content', '')

        if user_input:
            parsed_data.append((sys_prompt, user_input, gt_text))

    return parsed_data


def main():
    # 1. 指定 GGUF 模型文件与推理参数
    gguf_model_path = "/media/inno/work_dirs/LLM/LlamaFactory/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora/gguf/Qwen3.5-4B-Q8_0.gguf"
    output_dir = "/media/inno/output/LLM/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora/"
    os.makedirs(output_dir, exist_ok=True)

    val_json_path = os.path.join(output_dir, "val_q8_0.json")
    val_md_path = os.path.join(output_dir, "val_q8_0.md")

    n_ctx = 4096           # 上下文窗口大小
    n_gpu_layers = -1      # GPU 卸载层数，-1 表示全部层卸载至 GPU（若 GPU 显存足够）

    # 2. 加载 GGUF 模型
    print(f"正在通过 llama_cpp 加载 GGUF 模型: {gguf_model_path}...")
    start_init_time = time.time()

    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False  # 设置为 False 关闭 C++ 刷屏日志
    )

    end_init_time = time.time()
    init_duration = end_init_time - start_init_time
    print(f"GGUF 模型加载完成，初始化耗时: {init_duration:.2f} 秒。")

    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        print(f"模型加载后 PyTorch 侧 GPU 显存占用: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")

    # 3. 读取验证集评估数据
    val_file_path = '/media/inno/LLM/GI/TrainData/V2/sharegpt/val.jsonl'
    eval_samples = load_val_samples(val_file_path)
    print(f"总计成功解析待测样例数: {len(eval_samples)}")

    val_raw_results = {}
    eval_details = []
    total_generated_tokens = 0

    total_cer = 0.0
    total_acc = 0.0
    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0

    # 4. 循环进行 GGUF 推理与结果对比
    print("\n开始 GGUF 多任务推理与 GT 结果对比评估...")
    start_inference_time = time.time()

    for idx, (sys_prompt, user_input, gt_text) in enumerate(eval_samples, 1):
        sample_key = f"sample_{idx}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input}
        ]

        sample_start_time = time.time()

        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            repeat_penalty=1.1,
            max_tokens=2048,
        )

        pred_text = completion["choices"][0]["message"]["content"] or ""
        response_len = completion.get("usage", {}).get("completion_tokens", 0)
        total_generated_tokens += response_len

        # 保存原始 raw 输出到 val_q8_0.json
        val_raw_results[sample_key] = pred_text.strip()

        # 解析 think 与 output
        gt_think, gt_output = parse_think_and_output(gt_text)
        pred_think, pred_output = parse_think_and_output(pred_text)

        # 进一步分离 检查过程/镜检所见 和 检查结果/诊断结论
        gt_proc, gt_diag = extract_procedure_and_diagnosis(gt_output)
        pred_proc, pred_diag = extract_procedure_and_diagnosis(pred_output)

        # 计算 pred_output 与 gt_output 的对比指标
        cer, acc = compute_cer(gt_output, pred_output)
        prec, rec, f1 = compute_char_f1(gt_output, pred_output)

        total_cer += cer
        total_acc += acc
        total_p += prec
        total_r += rec
        total_f1 += f1

        eval_details.append({
            "sample_key": sample_key,
            "user_input": user_input,
            "gt_output": gt_output,
            "pred_output": pred_output,
            "gt_proc": gt_proc,
            "pred_proc": pred_proc,
            "gt_diag": gt_diag,
            "pred_diag": pred_diag,
            "cer": cer,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        # 控制台打印对比
        print(f"\n==================== 样例 [{idx}/{len(eval_samples)}] ====================")
        print(f"【输入指令 (User)】:\n{user_input}")
        print("-" * 50)
        print(f"【检查过程/镜检所见】:\n🟢 真实 (GT):\n{gt_proc}\n🔵 生成 (Pred):\n{pred_proc}")
        print("-" * 50)
        print(f"【诊断结论/检查结果】:\n🟢 真实 (GT):\n{gt_diag}\n🔵 生成 (Pred):\n{pred_diag}")
        print(f"【评估指标】: CER={cer:.4f} | Accuracy={acc:.4f} | F1={f1:.4f}")

        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time
        print(f"[耗时统计] 样本 {sample_key} GGUF 推理耗时: {sample_duration:.2f} 秒 (生成 {response_len} tokens)")
        print('=' * 60)

    end_inference_time = time.time()
    total_inference_duration = end_inference_time - start_inference_time

    # 5. 计算整体指标平均值
    num_samples = len(eval_samples)
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    avg_acc = total_acc / num_samples if num_samples > 0 else 0.0
    avg_p = total_p / num_samples if num_samples > 0 else 0.0
    avg_r = total_r / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0

    # 6. 获取 CPU 与内存指标
    cpu_percent = psutil.cpu_percent(interval=0.5)
    process = psutil.Process(os.getpid())
    process_cpu_percent = process.cpu_percent(interval=None)

    # 7. 打印最终性能与指标汇总报告
    print("\n================== GGUF 评估性能与准确率指标汇总 ==================")
    print(f"1. 评估样本数: {num_samples} 句")
    print(f"2. 平均 CER (字错率): {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
    print(f"3. 平均 字符准确率 (Accuracy): {avg_acc:.4f} ({avg_acc * 100:.2f}%)")
    print(f"4. 平均 字符 Precision / Recall / F1: {avg_p:.4f} / {avg_r:.4f} / {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")
    print("-" * 50)
    print(f"5. GGUF 模型加载耗时: {init_duration:.2f} 秒")
    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        print(f"6. GPU 显存占用量: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")
    print(f"7. 系统 CPU 占用比例: {cpu_percent:.1f}% (当前进程: {process_cpu_percent:.1f}%)")

    if num_samples > 0:
        avg_sample_time = total_inference_duration / num_samples
        avg_tokens_per_sec = total_generated_tokens / total_inference_duration if total_inference_duration > 0 else 0.0
        print(f"8. 总推理耗时: {total_inference_duration:.2f} 秒")
        print(f"9. 整体推理吞吐速度: {avg_tokens_per_sec:.2f} tokens/s")
        print(f"10. 平均单样本推理耗时: {avg_sample_time:.2f} 秒/样本")
    print("==================================================================\n")

    # 8. 保存 1: val_q8_0.json (格式："{case_no}": raw)
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_raw_results, f, ensure_ascii=False, indent=4)
    print(f"GGUF 模型原始预测结果已成功保存至 JSON: {val_json_path}")

    # 9. 保存 2: val_q8_0.md (对比评测报告与指标)
    md_lines = []
    md_lines.append("# GGUF GI 内镜报告生成模型评估对比报告 (val_q8_0.md)\n")
    md_lines.append("## 一、 整体评估指标汇总\n")
    md_lines.append(f"- **测试集样本数**: {num_samples} 句")
    md_lines.append(f"- **平均 CER (字错率)**: `{avg_cer:.4f}` ({avg_cer * 100:.2f}%)")
    md_lines.append(f"- **平均 字符准确率 (Accuracy)**: `{avg_acc:.4f}` ({avg_acc * 100:.2f}%)")
    md_lines.append(f"- **平均 字符 Precision**: `{avg_p:.4f}`")
    md_lines.append(f"- **平均 字符 Recall**: `{avg_r:.4f}`")
    md_lines.append(f"- **平均 字符 F1-Score**: `{avg_f1:.4f}` ({avg_f1 * 100:.2f}%)\n")
    md_lines.append(f"- **总推理耗时**: `{total_inference_duration:.2f}` 秒 (吞吐速度: `{avg_tokens_per_sec:.2f}` tokens/s)\n")

    md_lines.append("## 二、 逐样本报告结果对比\n")
    for idx, item in enumerate(eval_details, 1):
        md_lines.append(f"### 📌 样本 [{idx}/{num_samples}]: {item['sample_key']}\n")
        md_lines.append(f"> **评估指标**: **CER**: `{item['cer']:.4f}` | **Accuracy**: `{item['acc']:.4f}` | **F1-Score**: `{item['f1']:.4f}` (Precision: `{item['precision']:.4f}`, Recall: `{item['recall']:.4f}`)\n")
        md_lines.append(f"#### 🗣️ 输入口语描述 (User)\n```text\n{item['user_input']}\n```\n")

        # 检查过程 / 镜检所见 分开对比
        md_lines.append("#### 🔍 检查过程 / 镜检所见 对比\n")
        md_lines.append(f"**🟢 真实过程 (GT Process)**:\n```json\n{item['gt_proc'] if item['gt_proc'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 生成过程 (Pred Process)**:\n```json\n{item['pred_proc'] if item['pred_proc'] else '(无)'}\n```\n")

        # 诊断结论 / 检查结果 分开对比
        md_lines.append("#### 📋 诊断结论 / 检查结果 对比\n")
        md_lines.append(f"**🟢 真实诊断 (GT Diagnosis)**:\n```text\n{item['gt_diag'] if item['gt_diag'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 生成诊断 (Pred Diagnosis)**:\n```text\n{item['pred_diag'] if item['pred_diag'] else '(无)'}\n```\n")

        md_lines.append("---\n")

    with open(val_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"GGUF 模型评估对比报告已成功保存至 Markdown: {val_md_path}")


if __name__ == "__main__":
    main()
