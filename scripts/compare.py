import os
import json
import re
import argparse
from collections import Counter


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
    读取验证集文件，提取 user_input 与 gt_text
    """
    samples = []
    if not os.path.exists(val_file_path):
        alt_path = val_file_path.replace('.json', '.jsonl') if val_file_path.endswith('.json') else val_file_path.replace('.jsonl', '.json')
        if os.path.exists(alt_path):
            val_file_path = alt_path
        else:
            raise FileNotFoundError(f"未找到验证集文件: {val_file_path}")

    if val_file_path.endswith('.jsonl'):
        with open(val_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(val_file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

    parsed_data = []
    for item in samples:
        user_input = ""
        gt_text = ""

        if 'conversations' in item:
            for msg in item['conversations']:
                if msg.get('from') in ['human', 'user']:
                    user_input = msg.get('value', '')
                elif msg.get('from') in ['gpt', 'assistant']:
                    gt_text = msg.get('value', '')
        elif 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') in ['human', 'user']:
                    user_input = msg.get('content', '')
                elif msg.get('role') in ['gpt', 'assistant']:
                    gt_text = msg.get('content', '')

        if user_input:
            parsed_data.append((user_input, gt_text))

    return parsed_data


def load_model_predictions(model_dir: str):
    """
    读取模型输出目录下的 val_q8_0.json 或 val.json
    """
    json_path = os.path.join(model_dir, 'val_q8_0.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(model_dir, 'val.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"在目录 {model_dir} 中未找到 val_q8_0.json 或 val.json 文件！")

    print(f"正在读取模型预测结果: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="对比两个模型的生成结果与指标 (含 GT 对比)")
    parser.add_argument(
        "--dir_a",
        default="/media/inno/output/LLM/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora",
    )
    parser.add_argument(
        "--name_a",
        default="model_A",
        help="模型 A 展示名称"
    )
    parser.add_argument(
        "--dir_b",
        default="/media/inno/output/LLM/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora_rag",
        help="模型 B 预测输出结果目录 (优化模型)"
    )
    parser.add_argument(
        "--name_b",
        default="model_B",
        help="模型 B 展示名称"
    )
    parser.add_argument(
        "--val_file",
        default="/media/inno/LLM/GI/TrainData/V2/sharegpt/val.jsonl",
        help="验证集 ground truth 文件路径"
    )
    parser.add_argument(
        "--output_md",
        default="/media/inno/output/LLM/gi/compare.md",
        help="生成的对比 Markdown 报告保存路径"
    )

    args = parser.parse_args()

    # 1. 加载 GT 数据与模型 A/B 预测结果
    val_samples = load_val_samples(args.val_file)
    preds_a = load_model_predictions(args.dir_a)
    preds_b = load_model_predictions(args.dir_b)

    num_samples = len(val_samples)
    print(f"验证集样本数: {num_samples}, 模型 A 样例数: {len(preds_a)}, 模型 B 样例数: {len(preds_b)}")

    # 2. 统计指标与对比记录
    sample_details = []

    sum_cer_a, sum_acc_a, sum_p_a, sum_r_a, sum_f1_a = 0.0, 0.0, 0.0, 0.0, 0.0
    sum_cer_b, sum_acc_b, sum_p_b, sum_r_b, sum_f1_b = 0.0, 0.0, 0.0, 0.0, 0.0

    def get_str_text(val):
        if isinstance(val, dict):
            return val.get("raw_pred_text", "") or val.get("pred_output", "") or ""
        elif isinstance(val, str):
            return val
        return ""

    for idx, (user_input, gt_text) in enumerate(val_samples, 1):
        sample_key = f"sample_{idx}"
        
        # 提取真实答案 GT
        _, gt_output = parse_think_and_output(get_str_text(gt_text))
        gt_proc, gt_diag = extract_procedure_and_diagnosis(gt_output)

        # 提取模型 A 预测结果
        raw_text_a = get_str_text(preds_a.get(sample_key, ""))
        _, pred_output_a = parse_think_and_output(raw_text_a)
        proc_a, diag_a = extract_procedure_and_diagnosis(pred_output_a)
        cer_a, acc_a = compute_cer(gt_output, pred_output_a)
        prec_a, rec_a, f1_a = compute_char_f1(gt_output, pred_output_a)

        sum_cer_a += cer_a
        sum_acc_a += acc_a
        sum_p_a += prec_a
        sum_r_a += rec_a
        sum_f1_a += f1_a

        # 提取模型 B 预测结果
        raw_text_b = get_str_text(preds_b.get(sample_key, ""))
        _, pred_output_b = parse_think_and_output(raw_text_b)
        proc_b, diag_b = extract_procedure_and_diagnosis(pred_output_b)
        cer_b, acc_b = compute_cer(gt_output, pred_output_b)
        prec_b, rec_b, f1_b = compute_char_f1(gt_output, pred_output_b)

        sum_cer_b += cer_b
        sum_acc_b += acc_b
        sum_p_b += prec_b
        sum_r_b += rec_b
        sum_f1_b += f1_b

        sample_details.append({
            "sample_key": sample_key,
            "user_input": user_input,
            "gt_proc": gt_proc,
            "gt_diag": gt_diag,
            "proc_a": proc_a,
            "diag_a": diag_a,
            "cer_a": cer_a,
            "acc_a": acc_a,
            "f1_a": f1_a,
            "prec_a": prec_a,
            "rec_a": rec_a,
            "proc_b": proc_b,
            "diag_b": diag_b,
            "cer_b": cer_b,
            "acc_b": acc_b,
            "f1_b": f1_b,
            "prec_b": prec_b,
            "rec_b": rec_b,
        })

    # 3. 计算全局平均值
    avg_cer_a = sum_cer_a / num_samples if num_samples > 0 else 0.0
    avg_acc_a = sum_acc_a / num_samples if num_samples > 0 else 0.0
    avg_p_a = sum_p_a / num_samples if num_samples > 0 else 0.0
    avg_r_a = sum_r_a / num_samples if num_samples > 0 else 0.0
    avg_f1_a = sum_f1_a / num_samples if num_samples > 0 else 0.0

    avg_cer_b = sum_cer_b / num_samples if num_samples > 0 else 0.0
    avg_acc_b = sum_acc_b / num_samples if num_samples > 0 else 0.0
    avg_p_b = sum_p_b / num_samples if num_samples > 0 else 0.0
    avg_r_b = sum_r_b / num_samples if num_samples > 0 else 0.0
    avg_f1_b = sum_f1_b / num_samples if num_samples > 0 else 0.0

    # 4. 生成对比 Markdown 报告
    md_lines = []
    md_lines.append("# GI 内镜报告生成双模型效果对比评估报告\n")
    md_lines.append(f"- **模型 A (基线)**: `{args.name_a}`")
    md_lines.append(f"- **模型 B (优化)**: `{args.name_b}`")
    md_lines.append(f"- **验证集总样本数**: `{num_samples}` 句\n")

    md_lines.append("## 一、 整体评估指标对比汇总表\n")
    md_lines.append("| 评估指标 | 模型 A (基线) | 模型 B (优化) | 胜出模型 |")
    md_lines.append("| :--- | :---: | :---: | :---: |")

    winner_cer = "模型 B 🏆" if avg_cer_b < avg_cer_a else ("模型 A 🏆" if avg_cer_a < avg_cer_b else "平局")
    winner_acc = "模型 B 🏆" if avg_acc_b > avg_acc_a else ("模型 A 🏆" if avg_acc_a > avg_acc_b else "平局")
    winner_f1 = "模型 B 🏆" if avg_f1_b > avg_f1_a else ("模型 A 🏆" if avg_f1_a > avg_f1_b else "平局")

    md_lines.append(f"| **平均 CER (字错率)** | `{avg_cer_a:.4f}` ({avg_cer_a * 100:.2f}%) | `{avg_cer_b:.4f}` ({avg_cer_b * 100:.2f}%) | **{winner_cer}** |")
    md_lines.append(f"| **平均 字符准确率 (Accuracy)** | `{avg_acc_a:.4f}` ({avg_acc_a * 100:.2f}%) | `{avg_acc_b:.4f}` ({avg_acc_b * 100:.2f}%) | **{winner_acc}** |")
    md_lines.append(f"| **平均 字符 Precision** | `{avg_p_a:.4f}` | `{avg_p_b:.4f}` | - |")
    md_lines.append(f"| **平均 字符 Recall** | `{avg_r_a:.4f}` | `{avg_r_b:.4f}` | - |")
    md_lines.append(f"| **平均 字符 F1-Score** | `{avg_f1_a:.4f}` ({avg_f1_a * 100:.2f}%) | `{avg_f1_b:.4f}` ({avg_f1_b * 100:.2f}%) | **{winner_f1}** |\n")

    md_lines.append("---\n")
    md_lines.append("## 二、 逐样本三方对比 (GT vs 模型 A vs 模型 B)\n")

    for idx, item in enumerate(sample_details, 1):
        md_lines.append(f"### 📌 样本 [{idx}/{num_samples}]: {item['sample_key']}\n")
        md_lines.append(f"> **指标对比**:\n"
                        f"> - **模型 A (基线)**: CER = `{item['cer_a']:.4f}` | Accuracy = `{item['acc_a']:.4f}` | F1 = `{item['f1_a']:.4f}`\n"
                        f"> - **模型 B (优化)**: CER = `{item['cer_b']:.4f}` | Accuracy = `{item['acc_b']:.4f}` | F1 = `{item['f1_b']:.4f}`\n")

        md_lines.append(f"#### 🗣️ 输入口语描述 (User)\n```text\n{item['user_input']}\n```\n")

        # 三方 检查过程 / 镜检所见 对比
        md_lines.append("#### 🔍 检查过程 / 镜检所见 对比 (GT vs 模型 A vs 模型 B)\n")
        md_lines.append(f"**🟢 真实过程 (GT Process)**:\n```json\n{item['gt_proc'] if item['gt_proc'] else '(无)'}\n```\n")
        md_lines.append(f"**🟠 模型 A 过程 (Model A Process)**:\n```json\n{item['proc_a'] if item['proc_a'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 模型 B 过程 (Model B Process)**:\n```json\n{item['proc_b'] if item['proc_b'] else '(无)'}\n```\n")

        # 三方 诊断结论 / 检查结果 对比
        md_lines.append("#### 📋 诊断结论 / 检查结果 对比 (GT vs 模型 A vs 模型 B)\n")
        md_lines.append(f"**🟢 真实诊断 (GT Diagnosis)**:\n```text\n{item['gt_diag'] if item['gt_diag'] else '(无)'}\n```\n")
        md_lines.append(f"**🟠 模型 A 诊断 (Model A Diagnosis)**:\n```text\n{item['diag_a'] if item['diag_a'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 模型 B 诊断 (Model B Diagnosis)**:\n```text\n{item['diag_b'] if item['diag_b'] else '(无)'}\n```\n")

        md_lines.append("---\n")

    output_dir = os.path.dirname(args.output_md)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print("\n================== 整体对比评估结果 ==================")
    print(f"模型 A (基线): CER={avg_cer_a:.4f} (Acc={avg_acc_a:.4f}), F1={avg_f1_a:.4f}")
    print(f"模型 B (优化): CER={avg_cer_b:.4f} (Acc={avg_acc_b:.4f}), F1={avg_f1_b:.4f}")
    print(f"胜出模型: CER ({winner_cer}), F1 ({winner_f1})")
    print(f"对比 Markdown 报告已成功保存至: {args.output_md}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
