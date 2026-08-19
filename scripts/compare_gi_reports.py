#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消化内镜语音转报告双版本效果对比与 GT 评估脚本
功能:
1. 加载 GT 真实标注数据集 (sharegpt/val.jsonl);
2. 分别读取 gi_report 与 gi_report_0818 两个版本的生成报告;
3. 建立 1 对 1 精准映射，提取音频输入、ASR 转写、GT 报告与两版本模型输出;
4. 综合计算 CER (字错率)、Accuracy (字符准确率)、Precision、Recall、F1-Score;
5. 导出结构化详尽的 Markdown 对比报告。
"""

import os
import sys
import json
import re
import argparse
from collections import Counter


def compute_cer(gt_text: str, pred_text: str):
    """计算字符错误率 CER (Character Error Rate) 与 字符准确率 (Accuracy)"""
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
    """计算字符级 Precision, Recall, F1-Score"""
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


def parse_think_and_output(text: str):
    """解析文本中 <think>...</think> 思考过程与最终输出内容"""
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


def extract_report_fields(report_data):
    """从 report 字典或字符串中提取规整的 '镜检所见' 和 '诊断结论' 文本"""
    if isinstance(report_data, dict):
        proc = report_data.get("镜检所见") or report_data.get("检查过程") or ""
        diag = report_data.get("诊断结论") or report_data.get("检查结果") or ""

        if isinstance(proc, dict):
            proc_str = "\n".join([f"{k}：{v}" for k, v in proc.items()])
        elif isinstance(proc, list):
            proc_str = "\n".join([str(item) for item in proc])
        else:
            proc_str = str(proc)

        if isinstance(diag, list):
            diag_str = "；".join([str(item) for item in diag])
        else:
            diag_str = str(diag)

        return proc_str.strip(), diag_str.strip()

    if isinstance(report_data, str):
        # 尝试反序列化 JSON
        try:
            d = json.loads(report_data)
            if isinstance(d, dict):
                return extract_report_fields(d)
        except Exception:
            pass
        return report_data.strip(), ""

    return "", ""


def load_gt_val_samples(val_file_path: str):
    """读取验证集标注文件"""
    if not os.path.exists(val_file_path):
        raise FileNotFoundError(f"未找到验证集文件: {val_file_path}")

    samples = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                samples.append(json.loads(line_str))

    parsed = []
    for item in samples:
        user_input = ""
        gt_text = ""
        convs = item.get("conversations", [])
        if convs:
            for msg in convs:
                if msg.get("from") in ["human", "user"]:
                    user_input = msg.get("value", "")
                elif msg.get("from") in ["gpt", "assistant"]:
                    gt_text = msg.get("value", "")
        elif "messages" in item:
            for msg in item["messages"]:
                if msg.get("role") in ["human", "user"]:
                    user_input = msg.get("content", "")
                elif msg.get("role") in ["gpt", "assistant"]:
                    gt_text = msg.get("content", "")

        parsed.append({"user_input": user_input, "gt_raw": gt_text})
    return parsed


def load_output_dir(dir_path: str):
    """加载指定输出目录下的所有 *_report.json 文件"""
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"输出目录不存在: {dir_path}")

    reports = {}
    for fname in os.listdir(dir_path):
        if fname.endswith("_report.json"):
            key = fname.replace("_report.json", "")
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "r", encoding="utf-8") as fp:
                reports[key] = json.load(fp)
    return reports


def format_report_for_eval(proc: str, diag: str) -> str:
    """组合镜检所见与诊断结论为统一评估字符串"""
    parts = []
    if proc:
        parts.append(f"【镜检所见】\n{proc}")
    if diag:
        parts.append(f"【诊断结论】\n{diag}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="对比 gi_report 与 gi_report_0818 并关联 GT 音频与报告")
    parser.add_argument(
        "--dir_a",
        default="/media/inno/output/LLM/gi/gi_report",
        help="基线版本报告目录 (gi_report)"
    )
    parser.add_argument(
        "--name_a",
        default="gi_report (基线版本)",
        help="版本 A 展示名称"
    )
    parser.add_argument(
        "--dir_b",
        default="/media/inno/output/LLM/gi/gi_report_rag",
        help="对比版本报告目录 (gi_report_rag)"
    )
    parser.add_argument(
        "--name_b",
        default="gi_report_rag (RAG版本)",
        help="版本 B 展示名称"
    )
    parser.add_argument(
        "--val_file",
        default="/media/inno/LLM/GI/TrainData/V2/sharegpt/val.jsonl",
        help="GT 标注文件路径 (sharegpt/val.jsonl)"
    )
    parser.add_argument(
        "--output_md",
        default="/media/inno/output/LLM/gi/compare_gi_report_base-vs-rag.md",
        help="输出对比 Markdown 文件路径"
    )

    args = parser.parse_args()

    # 1. 加载所有数据
    print(f"正在加载验证集 GT 数据: {args.val_file}...")
    gt_samples = load_gt_val_samples(args.val_file)

    print(f"正在加载版本 A 报告: {args.dir_a}...")
    reps_a = load_output_dir(args.dir_a)

    print(f"正在加载版本 B 报告: {args.dir_b}...")
    reps_b = load_output_dir(args.dir_b)

    num_samples = len(gt_samples)
    print(f"GT 样本数: {num_samples}, 版本 A 报告数: {len(reps_a)}, 版本 B 报告数: {len(reps_b)}")

    # 2. 建立精准 1 对 1 匹配
    matched_records = []
    used_keys = set()

    for idx, gt_item in enumerate(gt_samples, 1):
        user_input = gt_item["user_input"]
        gt_raw = gt_item["gt_raw"]

        # 解析 GT 的 think 和输出
        _, gt_final = parse_think_and_output(gt_raw)
        gt_proc, gt_diag = extract_report_fields(gt_final)
        gt_eval_str = format_report_for_eval(gt_proc, gt_diag)

        clean_user = re.sub(r"^提取有效信息[^\n:：]+[：:]\s*", "", user_input).replace("？", "").replace(" ", "").replace("，", "").replace("。", "")

        best_key = None
        best_score = -1

        for key, rep in reps_b.items():
            if key in used_keys:
                continue
            asr = (rep.get("asr_corrected_text") or rep.get("asr_raw_text") or "").replace(" ", "").replace("，", "").replace("。", "")
            score = sum(1 for ch in set(clean_user) if ch in asr)
            if clean_user[:15] in asr or asr[:15] in clean_user:
                score += 50
            if score > best_score:
                best_score = score
                best_key = key

        if best_key:
            used_keys.add(best_key)
        else:
            # 兜底从未使用的 key 中选一个
            remains = [k for k in reps_b if k not in used_keys]
            best_key = remains[0] if remains else f"sample_{idx}"
            used_keys.add(best_key)

        item_a = reps_a.get(best_key, {})
        item_b = reps_b.get(best_key, {})

        audio_path = item_b.get("audio_path") or item_a.get("audio_path") or f"未知音频 ({best_key})"
        audio_dur = item_b.get("audio_duration_sec") or item_a.get("audio_duration_sec") or 0.0
        gi_type = item_b.get("gi_type_tag") or item_a.get("gi_type_tag") or "gastro"

        asr_text_a = item_a.get("asr_corrected_text") or item_a.get("asr_raw_text") or ""
        asr_text_b = item_b.get("asr_corrected_text") or item_b.get("asr_raw_text") or ""

        # 提取 A 报告字段
        rep_a_obj = item_a.get("report")
        if isinstance(rep_a_obj, dict):
            proc_a, diag_a = extract_report_fields(rep_a_obj)
        else:
            _, out_a = parse_think_and_output(item_a.get("report_output", ""))
            proc_a, diag_a = extract_report_fields(out_a)
        eval_str_a = format_report_for_eval(proc_a, diag_a)

        # 提取 B 报告字段
        rep_b_obj = item_b.get("report")
        if isinstance(rep_b_obj, dict):
            proc_b, diag_b = extract_report_fields(rep_b_obj)
        else:
            _, out_b = parse_think_and_output(item_b.get("report_output", ""))
            proc_b, diag_b = extract_report_fields(out_b)
        eval_str_b = format_report_for_eval(proc_b, diag_b)

        # 计算指标
        cer_a, acc_a = compute_cer(gt_eval_str, eval_str_a)
        prec_a, rec_a, f1_a = compute_char_f1(gt_eval_str, eval_str_a)

        cer_b, acc_b = compute_cer(gt_eval_str, eval_str_b)
        prec_b, rec_b, f1_b = compute_char_f1(gt_eval_str, eval_str_b)

        # 单独针对诊断结论的指标
        diag_cer_a, diag_acc_a = compute_cer(gt_diag, diag_a)
        diag_f1_a, _, _ = compute_char_f1(gt_diag, diag_a)
        diag_cer_b, diag_acc_b = compute_cer(gt_diag, diag_b)
        diag_f1_b, _, _ = compute_char_f1(gt_diag, diag_b)

        matched_records.append({
            "idx": idx,
            "sample_key": best_key,
            "audio_path": audio_path,
            "audio_duration_sec": audio_dur,
            "gi_type": gi_type,
            "user_prompt": user_input,
            "asr_text_a": asr_text_a,
            "asr_text_b": asr_text_b,
            "gt_proc": gt_proc,
            "gt_diag": gt_diag,
            "gt_eval_str": gt_eval_str,
            "proc_a": proc_a,
            "diag_a": diag_a,
            "eval_str_a": eval_str_a,
            "cer_a": cer_a,
            "acc_a": acc_a,
            "f1_a": f1_a,
            "prec_a": prec_a,
            "rec_a": rec_a,
            "diag_cer_a": diag_cer_a,
            "diag_acc_a": diag_acc_a,
            "diag_f1_a": diag_f1_a,
            "proc_b": proc_b,
            "diag_b": diag_b,
            "eval_str_b": eval_str_b,
            "cer_b": cer_b,
            "acc_b": acc_b,
            "f1_b": f1_b,
            "prec_b": prec_b,
            "rec_b": rec_b,
            "diag_cer_b": diag_cer_b,
            "diag_acc_b": diag_acc_b,
            "diag_f1_b": diag_f1_b,
        })

    # 3. 计算全局平均统计指标
    n = len(matched_records)
    avg_cer_a = sum(r["cer_a"] for r in matched_records) / n if n > 0 else 0
    avg_acc_a = sum(r["acc_a"] for r in matched_records) / n if n > 0 else 0
    avg_p_a = sum(r["prec_a"] for r in matched_records) / n if n > 0 else 0
    avg_r_a = sum(r["rec_a"] for r in matched_records) / n if n > 0 else 0
    avg_f1_a = sum(r["f1_a"] for r in matched_records) / n if n > 0 else 0

    avg_cer_b = sum(r["cer_b"] for r in matched_records) / n if n > 0 else 0
    avg_acc_b = sum(r["acc_b"] for r in matched_records) / n if n > 0 else 0
    avg_p_b = sum(r["prec_b"] for r in matched_records) / n if n > 0 else 0
    avg_r_b = sum(r["rec_b"] for r in matched_records) / n if n > 0 else 0
    avg_f1_b = sum(r["f1_b"] for r in matched_records) / n if n > 0 else 0

    # 诊断结论指标
    avg_diag_cer_a = sum(r["diag_cer_a"] for r in matched_records) / n if n > 0 else 0
    avg_diag_acc_a = sum(r["diag_acc_a"] for r in matched_records) / n if n > 0 else 0
    avg_diag_cer_b = sum(r["diag_cer_b"] for r in matched_records) / n if n > 0 else 0
    avg_diag_acc_b = sum(r["diag_acc_b"] for r in matched_records) / n if n > 0 else 0

    # 4. 生成对比 Markdown 报告
    winner_cer = f"{args.name_b} 🏆" if avg_cer_b < avg_cer_a else (f"{args.name_a} 🏆" if avg_cer_a < avg_cer_b else "平局")
    winner_acc = f"{args.name_b} 🏆" if avg_acc_b > avg_acc_a else (f"{args.name_a} 🏆" if avg_acc_a > avg_acc_b else "平局")
    winner_f1 = f"{args.name_b} 🏆" if avg_f1_b > avg_f1_a else (f"{args.name_a} 🏆" if avg_f1_a > avg_f1_b else "平局")
    winner_diag_acc = f"{args.name_b} 🏆" if avg_diag_acc_b > avg_diag_acc_a else (f"{args.name_a} 🏆" if avg_diag_acc_a > avg_diag_acc_b else "平局")

    md = []
    md.append("# 消化内镜语音转报告双版本 (gi_report vs gi_report_0818) 全面评测与三方对比报告\n")
    md.append(f"- **对比版本 A (基线)**: `{args.dir_a}` (`{args.name_a}`)")
    md.append(f"- **对比版本 B (优化)**: `{args.dir_b}` (`{args.name_b}`)")
    md.append(f"- **Ground Truth (GT 真实标准)**: `{args.val_file}`")
    md.append(f"- **评测验证集样本数**: `{num_samples}` 个音频与报告用例\n")

    md.append("## 一、 全局评测指标汇总表\n")
    md.append("| 评测维度 / 指标 | 版本 A (`gi_report`) | 版本 B (`gi_report_0818`) | 差异变化 (B - A) | 胜出版本 |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")

    cer_diff = avg_cer_b - avg_cer_a
    acc_diff = avg_acc_b - avg_acc_a
    f1_diff = avg_f1_b - avg_f1_a
    diag_acc_diff = avg_diag_acc_b - avg_diag_acc_a

    md.append(f"| **全报告平均 CER (字错率)** | `{avg_cer_a:.4f}` ({avg_cer_a * 100:.2f}%) | `{avg_cer_b:.4f}` ({avg_cer_b * 100:.2f}%) | `{cer_diff:+.4f}` ({cer_diff * 100:+.2f}%) | **{winner_cer}** |")
    md.append(f"| **全报告平均 Accuracy (准确率)** | `{avg_acc_a:.4f}` ({avg_acc_a * 100:.2f}%) | `{avg_acc_b:.4f}` ({avg_acc_b * 100:.2f}%) | `{acc_diff:+.4f}` ({acc_diff * 100:+.2f}%) | **{winner_acc}** |")
    md.append(f"| **全报告平均 F1-Score** | `{avg_f1_a:.4f}` ({avg_f1_a * 100:.2f}%) | `{avg_f1_b:.4f}` ({avg_f1_b * 100:.2f}%) | `{f1_diff:+.4f}` ({f1_diff * 100:+.2f}%) | **{winner_f1}** |")
    md.append(f"| **全报告平均 Precision** | `{avg_p_a:.4f}` | `{avg_p_b:.4f}` | `{avg_p_b - avg_p_a:+.4f}` | - |")
    md.append(f"| **全报告平均 Recall** | `{avg_r_a:.4f}` | `{avg_r_b:.4f}` | `{avg_r_b - avg_r_a:+.4f}` | - |")
    md.append(f"| **诊断结论平均 Accuracy (准确率)** | `{avg_diag_acc_a:.4f}` ({avg_diag_acc_a * 100:.2f}%) | `{avg_diag_acc_b:.4f}` ({avg_diag_acc_b * 100:.2f}%) | `{diag_acc_diff:+.4f}` ({diag_acc_diff * 100:+.2f}%) | **{winner_diag_acc}** |\n")

    md.append("---\n")
    md.append("## 二、 逐样本全流程三方对比 (音频输入 + GT标准 vs gi_report vs gi_report_0818)\n")

    for r in matched_records:
        idx = r["idx"]
        key = r["sample_key"]
        audio_path = r["audio_path"]
        dur = r["audio_duration_sec"]
        gi_type = r["gi_type"]

        md.append(f"### 📌 样本 [{idx}/{n}]: `{key}`")
        md.append(f"- **音频文件路径**: `{audio_path}` (时长: `{dur:.2f}s` | 类型: `{gi_type}`)")
        md.append(f"- **单样本指标对比**:")
        md.append(f"  - **版本 A (gi_report)**: CER = `{r['cer_a']:.4f}` | Accuracy = `{r['acc_a']:.4f}` | F1 = `{r['f1_a']:.4f}`")
        md.append(f"  - **版本 B (gi_report_0818)**: CER = `{r['cer_b']:.4f}` | Accuracy = `{r['acc_b']:.4f}` | F1 = `{r['f1_b']:.4f}`\n")

        md.append("#### 🎙️ 音频输入与口语转写 (ASR & User Prompt)")
        md.append(f"**🟢 GT 真实标准口语输入 (User Prompt)**:\n```text\n{r['user_prompt']}\n```")
        md.append(f"**🟠 版本 A ASR 识别转写**:\n```text\n{r['asr_text_a']}\n```")
        md.append(f"**🔵 版本 B ASR 识别转写**:\n```text\n{r['asr_text_b']}\n```\n")

        md.append("#### 🔍 镜检所见 / 检查过程 三方对比")
        md.append(f"**🟢 GT 真实镜检所见 (Ground Truth)**:\n```text\n{r['gt_proc'] if r['gt_proc'] else '(无)'}\n```")
        md.append(f"**🟠 版本 A 镜检所见 (gi_report)**:\n```text\n{r['proc_a'] if r['proc_a'] else '(无)'}\n```")
        md.append(f"**🔵 版本 B 镜检所见 (gi_report_0818)**:\n```text\n{r['proc_b'] if r['proc_b'] else '(无)'}\n```\n")

        md.append("#### 📋 诊断结论 三方对比")
        md.append(f"**🟢 GT 真实诊断结论 (Ground Truth)**:\n```text\n{r['gt_diag'] if r['gt_diag'] else '(无)'}\n```")
        md.append(f"**🟠 版本 A 诊断结论 (gi_report)**:\n```text\n{r['diag_a'] if r['diag_a'] else '(无)'}\n```")
        md.append(f"**🔵 版本 B 诊断结论 (gi_report_0818)**:\n```text\n{r['diag_b'] if r['diag_b'] else '(无)'}\n```\n")

        md.append("---\n")

    out_dir = os.path.dirname(args.output_md)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n" + "=" * 60)
    print("🎉 消化内镜语音转报告双版本对比完成！")
    print(f"📊 样本总数: {n}")
    print(f"🏆 全报告 CER: 版本 A={avg_cer_a:.4f} vs 版本 B={avg_cer_b:.4f} ({winner_cer})")
    print(f"🏆 全报告 Acc: 版本 A={avg_acc_a:.4f} vs 版本 B={avg_acc_b:.4f} ({winner_acc})")
    print(f"🏆 全报告 F1:  版本 A={avg_f1_a:.4f} vs 版本 B={avg_f1_b:.4f} ({winner_f1})")
    print(f"🏆 诊断 Acc:  版本 A={avg_diag_acc_a:.4f} vs 版本 B={avg_diag_acc_b:.4f} ({winner_diag_acc})")
    print(f"📄 详尽 Markdown 对比报告已保存至: {args.output_md}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
