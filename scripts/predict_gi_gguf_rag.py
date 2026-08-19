import os
import sys
import json
import re
import time
import psutil
import torch
import traceback
from collections import Counter

# 1. 尝试导入 llama_cpp 模块
try:
    from llama_cpp import Llama
except ImportError as e:
    print("\n" + "=" * 60)
    print("[导入错误] 无法导入 llama_cpp 模块！")
    print("错误详情:")
    traceback.print_exc()
    print("=" * 60)
    print("\n请检查是否在正确环境安装了 llama-cpp-python。")
    sys.exit(1)


# 2. 医疗精准规则路由器模块（聚焦 rules.json 精准病变实体匹配与纯净规则注入）
class MedicalRAGRetriever:
    def __init__(self, kb_dir: str = "/media/inno/LLM/RAG_knowledeg", bge_path: str = None):
        self.kb_dir = kb_dir
        self.rules = []
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        """初始化医疗规则知识库"""
        rules_path = os.path.join(self.kb_dir, "rules.json")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            print(f"成功加载医疗标准规范规则库 ({len(self.rules)} 条): {rules_path}")
        else:
            print(f"⚠️ 规则库文件不存在: {rules_path}")

    def retrieve(self, query: str, sys_prompt: str = "", top_k: int = 4) -> str:
        """按标点切分子句并结合解剖器官层级，精准绑定各病变实体的局域解剖部位，防止多部位跨句干扰与错配"""
        if not self.rules:
            return ""

        # 1. 判定检查类型 (胃镜 vs 肠镜硬隔离)
        is_gastro = "胃镜" in sys_prompt or "胃" in query or "食管" in query or "十二指肠" in query
        gi_target = "胃镜检查" if is_gastro else "肠镜检查"
        candidate_rules = [r for r in self.rules if r.get('gi') == gi_target or not r.get('gi')]

        matched_rules = []

        # 2. 特殊重要分型/标准强匹配逻辑 (支持单词触发与同子句多词组合 AND 触发)
        # 数据结构: (单词OR触发列表, 同子句多词AND组合列表, 目标规则Title)
        special_specs = [
            (
                ["木村", "竹本", "萎缩", "C1", "C2", "C3", "O1", "O2", "O3", "以白为主", "血管透见", "皱襞变平", "皱襞消失"], 
                [], 
                "萎缩性胃炎"
            ),
            (
                ["Boston", "波士顿", "BBPS", "清洁度", "粪水", "粪渣"], 
                [], 
                "Boston"
            ),
            (
                ["反流", "洛杉矶", "LA-", "A级", "B级", "C级", "D级"], 
                [["食管", "糜烂"], ["齿状线", "糜烂"], ["食管", "破损"], ["齿状线", "破损"]], 
                "反流性食管炎"
            ),
            (
                ["Barrett", "巴雷特", "齿状线上移", "舌型", "全周型", "岛型"], 
                [["齿状线", "上移"]], 
                "Barrett"
            ),
            # 1) 明确癌变与明显占位 -> 匹配【食管癌 / 胃癌】
            (
                [ "食管癌", "食管占位", "蕈伞型", "溃疡型", "溃疡浸润型", "弥漫浸润型"], 
                [["食管", "肿物"], ["食管", "占位"], ["食管", "B3"], ["IPCL", "B3"]], 
                "食管癌"
            ),
            (
                ["胃癌", "胃占位", "Borrmann", "鲍尔曼", "博尔曼", "胃癌占位"], 
                [["胃", "肿物"], ["胃", "占位"]], 
                "胃癌"
            ),
            (
                ["结肠癌", "结肠占位", "蕈伞型", "溃疡型", "溃疡浸润型", "弥漫浸润型"], 
                [["结肠", "肿物"], ["结肠", "占位"]], 
                "结肠癌"
            ),
            # 2) 早期病变/上皮内瘤变/异型增生 -> 精准匹配【食管黏膜病变 / 胃黏膜病变】
            (
                ["食管癌", "IPCL", "AVA", "食管占位", "碘染"], 
                [["食管", "病变"], ["食管", "病灶"], ["食管", "茶褐色"], ["食管", "粗糙"], ["食管", "B1"], ["食管", "B2"], ["IPCL", "B1"], ["IPCL", "B2"], ["食管", "低级别上皮内瘤变"], ["食管", "高级别上皮内瘤变"], ["食管", "LGIN"], ["食管", "HGIN"], ["食管", "异型增生"], ['碘染', '不染'], ['碘染', '淡染']], 
                "食管黏膜病变"
            ),
            (
                ["DL", "IMVP", "IMSP", "微表面", "微血管", "VS"], 
                [["胃", "病变"], ["胃", "病灶"], ["胃", "茶褐色"], ["胃", "低级别上皮内瘤变"], ["胃", "高级别上皮内瘤变"], ["胃", "LGIN"], ["胃", "HGIN"], ["胃", "异型增生"], ['靛胭脂', '边界清晰'], ['醋酸', '紊乱']], 
                "胃黏膜病变"
            ),
            (
                ["平滑肌瘤"], 
                [], 
                "食管黏膜下隆起"
            ),
            (
                ["间质瘤"], 
                [], 
                "胃黏膜下隆起"
            ),
            (
                ["霉菌", "真菌", "念珠菌", "细胞刷"], 
                [["食管", "白色","附着物"], ["食管", "白斑"], ["食管", "白苔"]], 
                "霉菌性食管炎"
            ),
            (
                ["串珠状", "结节状"], 
                [["食管", "静脉曲张"], ["食管", "曲张静脉"]], 
                "食管静脉曲张"
            ),
            (
                [], 
                [["食管", "斑驳"], ["食管", "花斑样改变"]], 
                "斑驳食管"
            ),
            (
                ["食管裂孔疝", "疝囊", '滑动型疝', '食管旁疝', '混合型疝', '巨大疝'], 
                [], 
                "食管裂孔疝"
            ),
            (
                [], 
                [["胃", "静脉曲张"], ["胃", "曲张静脉"]], 
                "胃静脉曲张"
            ),
            (
                ["异位"], 
                [["食管", "橘红色"]], 
                "食管胃黏膜异位"
            ),
            (
                ["马赛克", "蛇皮样"],
                [],
                "门脉高压性胃病"
            ),
            (
                ["HP", "幽门螺旋杆菌", "现症感染"],
                [],
                "HP现症感染"
            )
        ]

        # 使用大分句 (以句号、分号、换行分隔) 判断多词组合 AND 条件
        query_lower = query.lower()
        major_clauses = re.split(r'[。；;!\?\n]+', query_lower)

        for single_kws, combo_kws_list, target_title in special_specs:
            is_hit = False

            # 1) 单词 OR 触发 (统一忽略大小写)
            if any(kw.lower() in query_lower for kw in single_kws):
                is_hit = True

            # 2) 同主句内的多词组合 AND 触发 (统一忽略大小写)
            if not is_hit and combo_kws_list:
                for mc in major_clauses:
                    for combo in combo_kws_list:
                        if all(ckw.lower() in mc for ckw in combo):
                            is_hit = True
                            break
                    if is_hit:
                        break

            if is_hit:
                for r in candidate_rules:
                    title_name = r.get("title", "")
                    cat_name = r.get("category", "")
                    # 拦截: 排除“非萎缩性胃炎”中包含“萎缩性胃炎”的子串误杀
                    if (target_title in title_name or target_title in cat_name) and f"非{target_title}" not in title_name:
                        if r not in matched_rules:
                            matched_rules.append(r)
                            break

        # 辅助函数：切分子句，精准获取 target_kw 所在子句（或向前最近子句）的解剖部位
        def get_clause_location(q_text: str, target_kw: str) -> str:
            all_locs = ["十二指肠", "食管", "贲门", "胃底", "胃体", "胃角", "胃窦", "幽门", "回肠", "结肠", "直肠", "盲肠", "肛周"]
            clauses = re.split(r'[。；;!\?\n，,]+', q_text)
            hit_idx = -1
            for idx, c in enumerate(clauses):
                if target_kw in c:
                    hit_idx = idx
                    break
            if hit_idx != -1:
                # 1. 优先在当前相同的子句内部寻找解剖部位
                for loc in all_locs:
                    if loc in clauses[hit_idx]:
                        return loc
                # 2. 当前子句若无部位描述，向前逆向最近子句寻找部位
                for idx in range(hit_idx - 1, -1, -1):
                    for loc in all_locs:
                        if loc in clauses[idx]:
                            return loc
            return ""

        # 3. 病变实体多类别均衡与子句局域部位打分匹配
        entity_keywords = [
            "息肉", "溃疡", "黄色素瘤", "黄色瘤", "黏膜下隆起", "黏膜下肿瘤", "SMT", "静脉曲张", "静脉瘤", "乳头状瘤",
            "憩室", "狭窄", "糜烂", "平滑肌瘤", "十二指肠球炎", "十二指肠溃疡",
            "直肠炎", "结肠炎", "溃疡性结肠炎", "克罗恩", "锯齿状病变", "SSL", "侧向发育", "LST", "NICE", "Pit", "PP", "JNET", "山田"
        ]

        hit_entities = [kw for kw in entity_keywords if kw.lower() in query_lower]

        # 为每个命中的实体词挑选 1 条最符合其子句局域部位上下文的规则
        for kw in hit_entities:
            if len(matched_rules) >= top_k:
                break

            clause_loc = get_clause_location(query, kw)
            # 全量消化道器官层级推导归一化
            if clause_loc in ["食管上段", "食管中段", "食管下段", "门齿", "食管"]:
                loc_root = "食管"
            elif clause_loc in ["胃体", "胃窦", "胃角", "胃底", "贲门", "幽门", "胃"]:
                loc_root = "胃"
            elif clause_loc in ["十二指肠球部", "十二指肠降部", "球部", "降段", "十二指肠"]:
                loc_root = "十二指肠"
            elif clause_loc in ["盲肠", "升结肠", "横结肠", "降结肠", "乙状结肠", "回盲", "回盲部", "结肠"]:
                loc_root = "结肠"
            elif clause_loc in ["直肠", "肛周", "肛门"]:
                loc_root = "直肠"
            elif clause_loc in ["回肠末端", "末端回肠", "回肠"]:
                loc_root = "回肠"
            else:
                loc_root = clause_loc

            best_rule_for_kw = None
            best_score = -1

            for r in candidate_rules:
                if r in matched_rules:
                    continue

                title_name = r.get("title", "")
                cat_name = r.get("category", "")

                if kw in title_name or kw in cat_name:
                    score = 1
                    # 局域部位或大类层级匹配打分：优先奖励 title 直接匹配解剖大类 (+15分)，其次 category 匹配 (+10分)
                    if clause_loc and (clause_loc in title_name or (loc_root and loc_root in title_name)):
                        score += 15
                    elif clause_loc and (clause_loc in cat_name or (loc_root and loc_root in cat_name)):
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_rule_for_kw = r

            if best_rule_for_kw:
                matched_rules.append(best_rule_for_kw)

        # 4. 若无病变命中，回退基础兜底
        if not matched_rules:
            for r in candidate_rules:
                if "慢性浅表性胃炎" in r.get("title", "") or "未见明显异常" in r.get("title", ""):
                    matched_rules.append(r)
                    break

        # set_trace()
        # 5. 去重并限制不超过 top_k
        unique_rules = []
        for r in matched_rules:
            if r not in unique_rules:
                unique_rules.append(r)
            if len(unique_rules) >= top_k:
                break

        # 6. 格式化组装纯净 Prompt
        formatted_list = []
        for idx, r in enumerate(unique_rules, 1):
            cat_name = r.get("category", "")
            title_name = r.get("title", "")
            content_str = r.get("content", "").strip()
            formatted_list.append(f"• 规范{idx}【{cat_name} - {title_name}】:\n{content_str}")

        if formatted_list:
            return "【参考病变规范与标准模板】:\n" + "\n\n".join(formatted_list)
        return ""


def fix_boston_scores(text: str) -> str:
    """
    后处理纠错：确保输出 JSON 中 Boston 评分加和自洽 (Total = R + M + L)
    """
    pattern = r'总评分\s*(\d+)\s*分[，,]?\s*右半结肠[^\d]*(\d+)\s*分[，,]?\s*横结肠[^\d]*(\d+)\s*分[，,]?\s*左半结肠[^\d]*(\d+)\s*分'
    def replace_fn(match):
        tot, r, m, l = map(int, match.groups())
        r_c = max(0, min(3, r))
        m_c = max(0, min(3, m))
        l_c = max(0, min(3, l))
        correct_tot = r_c + m_c + l_c
        return f"总评分{correct_tot}分，右半结肠（盲肠、升结肠）{r_c}分、横结肠{m_c}分、左半结肠（降结肠、乙状结肠、直肠）{l_c}分"
    
    return re.sub(pattern, replace_fn, text)


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


def extract_procedure_and_diagnosis(output_str: str):
    """自适应解析内镜报告 JSON 字符串"""
    output_str = (output_str or "").strip()
    procedure = ""
    diagnosis = ""

    try:
        data = json.loads(output_str)
        if isinstance(data, dict):
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
    """计算字符错误率 CER (Character Error Rate)"""
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


def load_val_samples(val_file_path: str):
    """读取验证集评估数据"""
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
    # 1. 指定 GGUF 模型文件与输出目录
    gguf_model_path = "/media/inno/work_dirs/LLM/LlamaFactory/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora/gguf/Qwen3.5-4B-Q8_0.gguf"
    output_dir = "/media/inno/output/LLM/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora_rag/"
    os.makedirs(output_dir, exist_ok=True)

    kb_dir_path = "/media/inno/LLM/RAG_knowledeg"
    bge_path = "/home/inno/.cache/modelscope/hub/models/BGE/bge-small-zh-v1.5"

    val_json_path = os.path.join(output_dir, "val_q8_0.json")
    val_md_path = os.path.join(output_dir, "val_q8_0.md")

    n_ctx = 4096
    n_gpu_layers = -1

    # 2. 初始化 3D RAG 检索模块
    print("\n================== 初始化 3D 医疗 RAG 检索系统 ==================")
    rag_retriever = MedicalRAGRetriever(kb_dir=kb_dir_path, bge_path=bge_path)

    # 3. 加载 GGUF 模型
    print(f"\n正在通过 llama_cpp 加载 GGUF 模型: {gguf_model_path}...")
    start_init_time = time.time()

    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        seed=42,
        verbose=False
    )

    end_init_time = time.time()
    init_duration = end_init_time - start_init_time
    print(f"GGUF 模型加载完成，初始化耗时: {init_duration:.2f} 秒。")

    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2
        max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"模型加载后 GPU 显存占用: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")

    # 4. 读取验证集评估数据
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

    # 5. 循环进行 RAG + GGUF 推理与评估
    print("\n开始 RAG 检索增强 + GGUF 推理与对比评估...")
    start_inference_time = time.time()

    for idx, (base_sys_prompt, user_input, gt_text) in enumerate(eval_samples, 1):
        sample_key = f"sample_{idx}"

        # 5.1 执行 RAG 语义检索（使用双通道混合检索提取 Top-4 精准匹配规范）
        retrieved_knowledge = rag_retriever.retrieve(user_input, sys_prompt=base_sys_prompt, top_k=4)
        
        # 组装包含 RAG 检索规范的纯净 System Prompt
        if retrieved_knowledge:
            system_prompt = f"{base_sys_prompt}\n\n{retrieved_knowledge}"
        else:
            system_prompt = base_sys_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        sample_start_time = time.time()

        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )

        raw_pred_text = completion["choices"][0]["message"]["content"] or ""
        response_len = completion.get("usage", {}).get("completion_tokens", 0)
        total_generated_tokens += response_len

        # 5.2 实施后处理数值校验修复
        pred_text = fix_boston_scores(raw_pred_text)

        # 解析 think 与 output
        gt_think, gt_output = parse_think_and_output(gt_text)
        pred_think, pred_output = parse_think_and_output(pred_text)

        # 分离 检查过程 与 诊断结论
        gt_proc, gt_diag = extract_procedure_and_diagnosis(gt_output)
        pred_proc, pred_diag = extract_procedure_and_diagnosis(pred_output)

        # 保存结果到 val_q8_0.json (包含输入指令与预测结果)
        val_raw_results[sample_key] = {
            "input": user_input,
            "retrieved_rule": retrieved_knowledge,
            "pred_output": pred_output,
            "raw_pred_text": pred_text.strip()
        }

        # 计算评估指标
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
            "retrieved_rule": retrieved_knowledge,
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

        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time

        print(f"\n==================== 样例 [{idx}/{len(eval_samples)}] ====================")
        print(f"【输入指令/口语描述】:\n{user_input}")
        print("-" * 50)
        print(f"【RAG 检索匹配到的参考规范】:\n{retrieved_knowledge}")
        print("-" * 50)
        print(f"【检查过程/镜检所见】:\n🟢 真实 (GT):\n{gt_proc}\n🔵 生成 (Pred):\n{pred_proc}")
        print("-" * 50)
        print(f"【诊断结论/检查结果】:\n🟢 真实 (GT):\n{gt_diag}\n🔵 生成 (Pred):\n{pred_diag}")
        print(f"【评估指标】: CER={cer:.4f} | Accuracy={acc:.4f} | F1={f1:.4f}")
        print(f"[耗时统计] 推理耗时: {sample_duration:.2f} 秒 (生成 {response_len} tokens)")
        print('=' * 60)

    end_inference_time = time.time()
    total_inference_duration = end_inference_time - start_inference_time

    # 6. 统计整体指标
    num_samples = len(eval_samples)
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    avg_acc = total_acc / num_samples if num_samples > 0 else 0.0
    avg_p = total_p / num_samples if num_samples > 0 else 0.0
    avg_r = total_r / num_samples if num_samples > 0 else 0.0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0

    avg_tokens_per_sec = total_generated_tokens / total_inference_duration if total_inference_duration > 0 else 0.0

    print("\n================== RAG + GGUF 评估性能与指标汇总 ==================")
    print(f"1. 评估样本数: {num_samples} 句")
    print(f"2. 平均 CER (字错率): {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
    print(f"3. 平均 字符准确率 (Accuracy): {avg_acc:.4f} ({avg_acc * 100:.2f}%)")
    print(f"4. 平均 字符 F1-Score: {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")
    print(f"5. 总推理耗时: {total_inference_duration:.2f} 秒 (吞吐速度: {avg_tokens_per_sec:.2f} tokens/s)")
    print("==================================================================\n")

    # 7. 保存 val_q8_0.json
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_raw_results, f, ensure_ascii=False, indent=4)
    print(f"RAG GGUF 推理结果已成功保存至 JSON: {val_json_path}")

    # 8. 保存 val_q8_0.md
    md_lines = []
    md_lines.append("# RAG 检索增强 + GGUF 内镜报告生成模型评估对比报告 (val_q8_0.md)\n")
    md_lines.append("## 一、 整体评估指标汇总\n")
    md_lines.append(f"- **测试集样本数**: {num_samples} 句")
    md_lines.append(f"- **平均 CER (字错率)**: `{avg_cer:.4f}` ({avg_cer * 100:.2f}%)")
    md_lines.append(f"- **平均 字符准确率 (Accuracy)**: `{avg_acc:.4f}` ({avg_acc * 100:.2f}%)")
    md_lines.append(f"- **平均 字符 F1-Score**: `{avg_f1:.4f}` ({avg_f1 * 100:.2f}%)\n")
    md_lines.append(f"- **总推理耗时**: `{total_inference_duration:.2f}` 秒 (吞吐速度: `{avg_tokens_per_sec:.2f}` tokens/s)\n")

    md_lines.append("## 二、 逐样本报告结果对比\n")
    for idx, item in enumerate(eval_details, 1):
        md_lines.append(f"### 📌 样本 [{idx}/{num_samples}]: {item['sample_key']}\n")
        md_lines.append(f"> **评估指标**: **CER**: `{item['cer']:.4f}` | **Accuracy**: `{item['acc']:.4f}` | **F1-Score**: `{item['f1']:.4f}`\n")
        md_lines.append(f"#### 🗣️ 输入口语描述 (User)\n```text\n{item['user_input']}\n```\n")
        md_lines.append(f"#### 📖 RAG 检索出的匹配规范\n```text\n{item['retrieved_rule']}\n```\n")

        md_lines.append("#### 🔍 检查过程 / 镜检所见 对比\n")
        md_lines.append(f"**🟢 真实过程 (GT Process)**:\n```json\n{item['gt_proc'] if item['gt_proc'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 生成过程 (Pred Process)**:\n```json\n{item['pred_proc'] if item['pred_proc'] else '(无)'}\n```\n")

        md_lines.append("#### 📋 诊断结论 / 检查结果 对比\n")
        md_lines.append(f"**🟢 真实诊断 (GT Diagnosis)**:\n```text\n{item['gt_diag'] if item['gt_diag'] else '(无)'}\n```\n")
        md_lines.append(f"**🔵 生成诊断 (Pred Diagnosis)**:\n```text\n{item['pred_diag'] if item['pred_diag'] else '(无)'}\n```\n")

        md_lines.append("---\n")

    with open(val_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"RAG GGUF 评估报告已成功保存至 Markdown: {val_md_path}")


if __name__ == "__main__":
    main()
