import argparse
import json
import glob
import os
import random
import re
import shutil
import ctypes
import tempfile
from typing import List
from pathlib import Path
from ipdb import set_trace
import soundfile as sf

# 自动绑定 CUDA 运行时库
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + os.environ.get("LD_LIBRARY_PATH", "")

import llama_cpp
import llama_cpp.mtmd_cpp as mtmd
from llama_cpp import Llama


class Qwen3ASR17BGGUFEngine:
    """
    Qwen3-ASR-1.7B 原生 GGUF (llama.cpp MTMD) 多模态语音转写引擎
    """
    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int = 4096):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到 LLM GGUF 模型: {model_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"未找到 Audio mmproj GGUF 模型: {mmproj_path}")

        print(f"1. 正在加载 Qwen3-ASR-1.7B LLM (llama_cpp GPU): {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False
        )

        print(f"2. 正在加载 Qwen3-ASR Audio Projector (mmproj): {mmproj_path}")
        ctx_params = mtmd.mtmd_context_params_default()
        ctx_params.use_gpu = True
        ctx_params.print_timings = False
        ctx_params.n_threads = 8

        self.mtmd_ctx = mtmd.mtmd_init_from_file(
            mmproj_path.encode('utf-8'),
            self.llm.model,
            ctx_params
        )

        if not self.mtmd_ctx:
            raise RuntimeError("加载 MTMD 音频编码上下文失败！")

    def transcribe(self, audio_path: str, prompt_text: str = "语音转写成中文：") -> str:
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self.llm._ctx.ctx), True)
        # 转换成 16kHz float32 音频数据
        tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav = tf.name
        tf.close()

        try:
            ffmpeg_cmd = f'ffmpeg -i "{audio_path}" -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_wav}" > /dev/null 2>&1'
            os.system(ffmpeg_cmd)

            wav_data, sr = sf.read(temp_wav, dtype='float32')
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)

            n_samples = len(wav_data)
            float_array_type = ctypes.c_float * n_samples
            float_array = float_array_type(*wav_data)

            bitmap = mtmd.mtmd_bitmap_init_from_audio(n_samples, float_array)
            if not bitmap:
                raise RuntimeError("声学特征初始化失败！")

            chunks = mtmd.mtmd_input_chunks_init()
            marker = mtmd.mtmd_default_marker().decode('utf-8')
            full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{marker}{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            input_text = mtmd.mtmd_input_text(
                text=full_prompt.encode('utf-8'),
                add_special=True,
                parse_special=True
            )
            bitmaps = (ctypes.c_void_p * 1)(bitmap)

            mtmd.mtmd_tokenize(
                self.mtmd_ctx,
                chunks,
                ctypes.byref(input_text),
                bitmaps,
                1
            )

            new_n_past_val = llama_cpp.llama_pos(0)
            mtmd.mtmd_helper_eval_chunks(
                self.mtmd_ctx,
                self.llm._ctx.ctx,
                chunks,
                0,
                0,
                2048,
                True,
                ctypes.byref(new_n_past_val)
            )

            # 自回归采样生成
            sp = llama_cpp.llama_sampler_chain_default_params()
            smpl = llama_cpp.llama_sampler_chain_init(sp)
            llama_cpp.llama_sampler_chain_add(smpl, llama_cpp.llama_sampler_init_greedy())

            generated_text = ""
            for _ in range(256):
                token = llama_cpp.llama_sampler_sample(smpl, self.llm._ctx.ctx, -1)
                if token in [self.llm.token_eos(), 151645]:  # <|im_end|>
                    break
                text_piece = self.llm.detokenize([token]).decode('utf-8', errors='ignore')
                generated_text += text_piece
                
                single_batch = llama_cpp.llama_batch_get_one(
                    (llama_cpp.llama_token * 1)(token),
                    1
                )
                if llama_cpp.llama_decode(self.llm._ctx.ctx, single_batch) != 0:
                    break

            llama_cpp.llama_sampler_free(smpl)
            mtmd.mtmd_bitmap_free(bitmap)
            mtmd.mtmd_input_chunks_free(chunks)

            # 清理特殊标签前缀
            clean_text = generated_text.replace("language Chinese", "").replace("<asr_text>", "").strip()
            return clean_text
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


def load_asr_model(asr_dir: str, llm_name: str = "Qwen3-ASR-1.7B-Q8_0.gguf", mmproj_name: str = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", n_ctx: int = 4096):
    """加载并初始化 Qwen3-ASR GGUF 模型"""
    print("加载 Qwen3-ASR GGUF 模型...")
    llm_path = os.path.join(asr_dir, llm_name)
    mmproj_path = os.path.join(asr_dir, mmproj_name)
    engine = Qwen3ASR17BGGUFEngine(model_path=llm_path, mmproj_path=mmproj_path, n_ctx=n_ctx)
    return engine


def load_corrector(hotwords_path: str, threshold: float = 0.85):
    """加载并初始化后处理医学纠错词表"""
    print("正在加载后处理医学纠错词表...")
    try:
        import sys
        sys.path.append('/home/inno/code/ASR/asr-hotword')
        from hotword import PhonemeCorrector
        corrector = PhonemeCorrector(threshold=threshold)
        if hotwords_path and os.path.exists(hotwords_path):
            with open(hotwords_path, "r", encoding="utf-8") as f:
                hotwords_content = f.read()
            corrector.update_hotwords(hotwords_content)
            print("纠错词表加载完成。")
            return corrector
        else:
            print(f"警告：未找到热词文件 {hotwords_path}")
            return None
    except Exception as e:
        print(f"初始化 PhonemeCorrector 后处理纠错模块失败: {e}")
        return None


def asr_transcribe(model, wav_path: str, corrector=None) -> str:
    raw_text = model.transcribe(wav_path)
    if corrector is not None:
        try:
            corrected_res = corrector.correct(raw_text)
            return corrected_res.text
        except Exception as e:
            print(f"后处理纠错执行失败: {e}")
    return raw_text

GASTRO_SYSTEM_PROMPT = "你是一个严谨的胃镜专家，请精准提取医生口语中的病变部位与特征描述，并结合标准胃镜规范生成结构化镜检所见与诊断结论，严禁漏诊与误诊。"
COLON_SYSTEM_PROMPT = "你是一个严谨的肠镜专家，请精准提取医生口语中的阳性病变与关键信息，并结合标准肠镜模板自动规范补充未见异常部位的阴性描述与诊断结论，严禁漏诊与误诊。"


def process_gastroscope(standard_file: str, oral_info: str, think_content: str = None) -> dict:
    """处理胃镜文本（支持 .txt 与 .json），解析并生成微调所需的规范提示和结论等"""
    if standard_file.endswith('.json'):
        with open(standard_file, 'r', encoding='utf-8') as f:
            raw_gastro = json.load(f)
        
        gastro_report = {}
        proc_val = raw_gastro.get("镜检所见") if "镜检所见" in raw_gastro else raw_gastro.get("检查过程", "")
        diag_val = raw_gastro.get("诊断结论") if "诊断结论" in raw_gastro else raw_gastro.get("检查结果", "")
        
        # 统一将【镜检所见】格式化为段落字符串 (txt 格式)，避免与肠镜的数据结构冲突
        if isinstance(proc_val, dict):
            lines = [f"{k}：{v}" for k, v in proc_val.items() if v]
            proc_val = "\n".join(lines)
        elif isinstance(proc_val, list):
            proc_val = "\n".join([str(x) for x in proc_val if x])

        gastro_report["镜检所见"] = proc_val
        gastro_report["诊断结论"] = diag_val
    else:
        assert False, f"文件 {standard_file} 格式不支持，请使用 JSON 格式"
    
    if isinstance(gastro_report.get('诊断结论'), list):
        gastro_report['诊断结论'] = '；'.join(gastro_report['诊断结论'])
    
    report_str = json.dumps(gastro_report, ensure_ascii=False)
    if think_content:
        # 去除已有的 <think> 和 </think> 标签以防重复，然后统一重新包裹
        think_content = re.sub(r'</?think>', '', think_content).strip()
        think_content = f"<think>\n{think_content}\n</think>"
        assistant_content = f"{think_content}\n{report_str}"
    else:
        assistant_content = f"{report_str}"
    
    message = {
        "messages": [
            {
                "role": "system",
                "content": GASTRO_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"提取有效信息,生成标准胃镜报告：{oral_info}？" 
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    return message


def process_colonscope(standard_file: str, oral_info: str, think_content: str = None) -> dict:
    """处理肠镜文本（支持 .txt 与 .json），直接生成内镜报告"""
    if standard_file.endswith('.json'):
        with open(standard_file, 'r', encoding='utf-8') as f:
            colon_report = json.load(f)
         
        # 确保【镜检所见】格式化为段落字符串 (txt 格式)
        proc_val = colon_report.get("镜检所见") if "镜检所见" in colon_report else colon_report.get("检查过程", "")
        if isinstance(proc_val, dict):
            lines = [f"{k}：{v}" for k, v in proc_val.items() if v]
            colon_report["镜检所见"] = "\n".join(lines)
        elif isinstance(proc_val, list):
            colon_report["镜检所见"] = "\n".join([str(x) for x in proc_val if x])
    else:
        with open(standard_file, 'r', encoding='utf-8') as f:
            standard_content = [line.strip() for line in f.readlines() if line.strip()]
        assert len(standard_content) == 4, f'肠镜标准文件行数不等于4: {standard_file}'

        colon_report = {
            '镜检所见': '',
            '诊断结论': '', 
        }
        for i, line in enumerate(standard_content):
            if i != 3:
                if colon_report['镜检所见']:
                    colon_report['镜检所见'] += '\n' + line
                else:
                    colon_report['镜检所见'] = line
            else:
                colon_report['诊断结论'] = re.split(f'[；;.。]', line.strip('\n'))
    
    if isinstance(colon_report.get('诊断结论'), list):
        colon_report['诊断结论'] = '；'.join(colon_report['诊断结论'])
    report_str = json.dumps(colon_report, ensure_ascii=False)
    if think_content:
        # 去除已有的 <think> 和 </think> 标签以防重复，然后统一重新包裹
        think_content = re.sub(r'</?think>', '', think_content).strip()
        think_content = f"<think>\n{think_content}\n</think>"
        assistant_content = f"{think_content}\n{report_str}"
    else:
        assistant_content = f"{report_str}"

    message = {
        "messages": [
            {
                "role": "system",
                "content": COLON_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"提取有效信息,生成标准肠镜报告：{oral_info}"
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    return message


def main():
    parser = argparse.ArgumentParser(description="LLM data ")
    # -------------------------------- ASR 配置 ----------------------------------------------------
    parser.add_argument('--asr_dir', default="/media/inno/work_dirs/ASR/qwen3-asr/qwen3-asr-1.7b-v4/gguf", help='Qwen3-ASR GGUF 模型目录')
    parser.add_argument('--llm_name', default="Qwen3-ASR-1.7B-Q8_0.gguf", help='LLM GGUF 文件名')
    parser.add_argument('--mmproj_name', default="mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", help='Audio Projector GGUF 文件名')
    parser.add_argument("--corrector_hotwords", default="/media/inno/ASR/gi_hotwords.txt", help="后处理医学纠错词表路径，可为空，词汇量<5000即可")
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--save_dir", default="/media/inno/LLM/GI/TrainData/V2/", help="训练数据集保存路径")
    parser.add_argument("--case_mode", default="/media/inno/ASR/TrainData/V4/case_mapping.json", help="JSON路径，包含train_cases和val_cases，用于指定病例划分集合")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------- 数据源配置区域 ------------------------------------------
    # ----------------(口语化文本/音频文件夹路径, 标准表达文件夹路径, think的思考链文件夹路径, 是否为口语化音频, 胃镜/肠镜)----------------
    data_sources = [
        # ----------------------------------------- V2-----------------------------------------
        ("/media/inno/ASR/胃镜/audio/train/real/case/", "/media/inno/ASR/胃镜/base_data/standard_2/case/", "/media/inno/ASR/胃镜/base_data/think/case/", True, 'gastro'),  # 胃镜数据
        ("/media/inno/ASR/肠镜/audio/train/real/case/", "/media/inno/ASR/肠镜/base_data/standard_1/case/", "/media/inno/ASR/肠镜/base_data/think/case/", True, 'colon'),  # 肠镜数据
    ]
    # ------------------------------------------------------------------------------------------

    assert len(data_sources) > 0, "请在代码的 data_sources 列表中至少配置一个数据源！"

    # 加载指定的病例划分
    train_cases = set()
    val_cases = set()
    if args.case_mode and os.path.exists(args.case_mode):
        try:
            with open(args.case_mode, "r", encoding="utf-8") as f:
                case_data = json.load(f)
                train_cases = set(case_data.get("train_cases", []))
                val_cases = set(case_data.get("val_cases", []))
            print(f"成功加载病例划分: train_cases {len(train_cases)} 个, val_cases {len(val_cases)} 个")
        except Exception as e:
            print(f"加载 case_mode 失败: {e}")

    # 判断是否需要加载 ASR 模型和纠错后处理
    any_asr = any(src[3] for src in data_sources)
    asr_model = None
    corrector = None
    if any_asr:
        asr_model = load_asr_model(args.asr_dir, llm_name=args.llm_name, mmproj_name=args.mmproj_name)
        corrector = load_corrector(args.corrector_hotwords, threshold=0.85)

    messages_dir = os.path.join(args.save_dir, 'messages')
    os.makedirs(messages_dir, exist_ok=True)

    # 清理已存在的 train.jsonl 和 val.jsonl 以免重复追加
    for mode in ['train', 'val']:
        mode_path = os.path.join(messages_dir, mode + '.jsonl')
        if os.path.exists(mode_path):
            os.remove(mode_path)

    # 用于保存病例号与口语化表达的对应字典
    train_ann = {}
    val_ann = {}

    # 用于保存病例的文件名 (case_name)，供 status.json 统计使用
    train_case_names = []
    val_case_names = []

    # 用于动态控制 val 比例不超过 10% 的全局计数器
    global_train_count = 0
    global_val_count = 0

    for oral_path, standard_path, think_path, asr, gi_type in data_sources:
        print(f"处理数据源: oral_path={oral_path}, standard_path={standard_path}, think_path={think_path}, asr={asr}")
        if not os.path.exists(standard_path):
            print(f"警告: 标准文件夹 {standard_path} 不存在，跳过该数据源")
            continue

        for lesion in os.listdir(standard_path):
            work_lesion_path = os.path.join(standard_path, lesion)
            if not os.path.isdir(work_lesion_path):
                continue

            standard_files = sorted(glob.glob(f'{work_lesion_path}/*.txt') + glob.glob(f'{work_lesion_path}/*.json'))
            for standard_file in standard_files:
                case_name = os.path.splitext(os.path.basename(standard_file))[0]

                # 获取 think 内容
                think_content = None
                if think_path and os.path.exists(think_path):
                    think_file = os.path.join(think_path, lesion, f"{case_name}.txt")
                    if os.path.exists(think_file):
                        with open(think_file, "r", encoding="utf-8") as f:
                            think_content = f.read().strip()

                # 判断当前病例为哪个集合 (优先用 case_mode 划分，其次采用动态限制保证 val 占比不超过 10%)
                if case_name in train_cases:
                    mode = 'train'
                elif case_name in val_cases:
                    mode = 'val'
                else:
                    # 否则根据目前 train 和 val 的数量，保证 val 不超过 10%
                    if (global_val_count + 1) / (global_train_count + global_val_count + 1) > 0.1:
                        mode = 'train'
                    else:
                        # 保证在满足上限的前提下，如果 val_count 仍为 0，强制设定一个 val，否则以 10% 概率划分
                        if global_val_count == 0 or random.random() <= 0.1:
                            mode = 'val'
                        else:
                            mode = 'train'

                # 更新全局计数器
                if mode == 'train':
                    global_train_count += 1
                else:
                    global_val_count += 1

                # 口语化描述句子读取与生成
                if asr:
                    # 确定对应 wav 音频路径，并断言其必须存在
                    wav_oral_file = standard_file.replace(standard_path, oral_path)
                    file_stem, file_ext = os.path.splitext(wav_oral_file)
                    wav_oral_txt = file_stem + '.txt'
                    wav_file = file_stem + '.wav'
                    assert os.path.exists(wav_file), f"wav音频文件不存在: {wav_file}"

                    if os.path.exists(wav_oral_txt):
                        with open(wav_oral_txt, 'r', encoding='utf-8') as f:
                            content = f.readlines()
                        assert len(content) == 1
                        oral_info = content[0].strip('\n')
                    else:
                        oral_info = asr_transcribe(asr_model, wav_file, corrector=corrector)
                        os.makedirs(os.path.dirname(wav_oral_txt), exist_ok=True)
                        with open(wav_oral_txt, 'w', encoding='utf-8') as f:
                            f.writelines(oral_info + '\n')

                    # 将口语化音频文件拷贝到 save_dir/audio/ 下的 train 或 val 子文件夹中
                    dest_audio_dir = os.path.join(args.save_dir, 'audio', mode, lesion)
                    os.makedirs(dest_audio_dir, exist_ok=True)
                    shutil.copy2(wav_file, os.path.join(dest_audio_dir, os.path.basename(wav_file)))
                else:
                    oral_file = os.path.splitext(standard_file.replace(standard_path, oral_path))[0] + '.txt'
                    assert os.path.exists(oral_file), f"口语化文本文件不存在: {oral_file}"
                    with open(oral_file, 'r', encoding='utf-8') as f:
                        oral_content = f.readlines()
                    assert len(oral_content) == 1
                    oral_info = oral_content[0].strip('\n')

                # 保存病例号与 oral_info 的映射关系
                if mode == 'train':
                    train_ann[case_name] = oral_info
                    train_case_names.append(case_name)
                else:
                    val_ann[case_name] = oral_info
                    val_case_names.append(case_name)

                # 根据内镜类型解析并生成微调消息结构
                if gi_type == "gastro":
                    message = process_gastroscope(standard_file, oral_info, think_content)
                elif gi_type == "colon":
                    message = process_colonscope(standard_file, oral_info, think_content)
                else:
                    raise ValueError(f"不支持的内镜类型: {gi_type}")

                mode_path = os.path.join(messages_dir, mode + '.jsonl')
                with open(mode_path, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(message, ensure_ascii=False) + '\n')

    # 将不同数据集的病例号跟 oral_info 对应保存为 train.json 和 val.json
    ann_dir = os.path.join(args.save_dir, 'ann')
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_ann, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ann_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_ann, f, ensure_ascii=False, indent=4)
    
    # 生成 status.json (记录 train_cases 和 val_cases 的具体值以及数量)
    status_data = {
        "train_cases": sorted(list(set(train_case_names))),
        "val_cases": sorted(list(set(val_case_names))),
        "train_count": len(train_case_names),
        "val_count": len(val_case_names)
    }
    status_path = os.path.join(args.save_dir, 'status.json')
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=4)
    
    print(f"数据处理完毕。训练集样本数: {global_train_count}, 验证集样本数: {global_val_count}")
    print(f"标注文件已保存至: {ann_dir}")
    print(f"病例划分状态汇总已保存至: {status_path}")


if __name__ == '__main__':
    random.seed(20260528)
    main()