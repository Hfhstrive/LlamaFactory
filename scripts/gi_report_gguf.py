#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消化内镜端到端语音转报告流水线 (GGUF 本地推理脚本)
功能:
1. 接收音频/视频文件或文件夹路径 (支持 wav, mp3, m4a, flac, aac, mp4 等)；
2. 加载 Qwen3-ASR 1.7B GGUF 原生多模态模型进行转码与语音识别 (完全对齐 test_gguf.py)；
3. ASR 后处理: 可选拼音同音字/医学热词校正 (PhonemeCorrector)；
4. 智能提取与标定检查类型标签 (gastro: 胃镜 / colon: 肠镜 / all: 双镜)，并注入专属专家 System Prompt；
5. 调用 LoRA 微调后的 Qwen3.5-4B GGUF 大模型生成结构化镜检所见与诊断结论；
6. 后处理: BBPS 波士顿评分自动化校验与医学热词二重纠错，结果批量保存导出为 JSON。
"""

import os
import sys
import time
import json
import ctypes
import re
import argparse
import tempfile
import subprocess
import warnings
from typing import Optional, Any

warnings.filterwarnings("ignore")
import soundfile as sf

import llama_cpp
import llama_cpp.mtmd_cpp as mtmd
from llama_cpp import Llama

# 动态引入 asr-hotword 项目路径
hotword_project_dir = '/home/inno/code/ASR/asr-hotword'
if os.path.exists(hotword_project_dir) and hotword_project_dir not in sys.path:
    sys.path.append(hotword_project_dir)

# 尝试导入拼音同音字后处理工具
try:
    from hotword import PhonemeCorrector
except ImportError:
    PhonemeCorrector = None


class MedicalRAGRetriever:
    """消化内镜 RAG 医疗检索增强器 (完全对齐 predict_gi_gguf_rag.py)"""
    def __init__(self, kb_dir: str = "/media/inno/LLM/RAG_knowledeg"):
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
            (
                ["食管癌", "食管占位", "蕈伞型", "溃疡型", "溃疡浸润型", "弥漫浸润型"], 
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
            (
                ["食管癌", "IPCL", "AVA", "食管占位", "碘染"], 
                [["食管", "病变"], ["食管", "病灶"], ["食管", "茶褐色"], ["食管", "粗糙"], ["食管", "B1"], ["食管", "B2"], ["IPCL", "B1"], ["IPCL", "B2"], ["食管", "低级别上皮内瘤变"], ["食管", "高级别上皮内瘤变"], ["食管", "LGIN"], ["食管", "HGIN"], ["食管", "异型增生"], ['碘染', '不染'], ['碘染', '淡染']], 
                "食管黏膜病变"
            ),
            (
                ["DL", "IMVP", "IMSP", "微表面", "微血管", "VS"], 
                [["胃", "病变"], ["胃", "病灶"],["胃", "茶褐色"], ["胃", "低级别上皮内瘤变"], ["胃", "高级别上皮内瘤变"], ["胃", "LGIN"], ["胃", "HGIN"], ["胃", "异型增生"], ['靛胭脂', '边界清晰'], ['醋酸', '紊乱']], 
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
                [["食管", "白色","附着物"], ["食管", "白斑"]], 
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

        query_lower = query.lower()
        major_clauses = re.split(r'[。；;!\?\n]+', query_lower)

        for single_kws, combo_kws_list, target_title in special_specs:
            is_hit = False

            if any(kw.lower() in query_lower for kw in single_kws):
                is_hit = True

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
                    if (target_title in title_name or target_title in cat_name) and f"非{target_title}" not in title_name:
                        if r not in matched_rules:
                            matched_rules.append(r)
                            break

        def get_clause_location(q_text: str, target_kw: str) -> str:
            all_locs = ["十二指肠", "食管", "贲门", "胃底", "胃体", "胃角", "胃窦", "幽门", "回肠", "结肠", "直肠", "盲肠", "肛周"]
            clauses = re.split(r'[。；;!\?\n，,]+', q_text)
            hit_idx = -1
            for idx, c in enumerate(clauses):
                if target_kw in c:
                    hit_idx = idx
                    break
            if hit_idx != -1:
                for loc in all_locs:
                    if loc in clauses[hit_idx]:
                        return loc
                for idx in range(hit_idx - 1, -1, -1):
                    for loc in all_locs:
                        if loc in clauses[idx]:
                            return loc
            return ""

        entity_keywords = [
            "息肉", "溃疡", "黄色素瘤", "黄色瘤", "黏膜下隆起", "黏膜下肿瘤", "SMT", "静脉曲张", "静脉瘤", "乳头状瘤",
            "憩室", "狭窄", "糜烂", "平滑肌瘤", "十二指肠球炎", "十二指肠溃疡",
            "直肠炎", "结肠炎", "溃疡性结肠炎", "克罗恩", "锯齿状病变", "SSL", "侧向发育", "LST", "NICE", "Pit", "PP", "JNET", "山田"
        ]

        hit_entities = [kw for kw in entity_keywords if kw.lower() in query_lower]

        for kw in hit_entities:
            if len(matched_rules) >= top_k:
                break

            clause_loc = get_clause_location(query, kw)
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
                    if clause_loc and (clause_loc in title_name or (loc_root and loc_root in title_name)):
                        score += 15
                    elif clause_loc and (clause_loc in cat_name or (loc_root and loc_root in cat_name)):
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_rule_for_kw = r

            if best_rule_for_kw:
                matched_rules.append(best_rule_for_kw)

        if not matched_rules:
            for r in candidate_rules:
                if "慢性浅表性胃炎" in r.get("title", "") or "未见明显异常" in r.get("title", ""):
                    matched_rules.append(r)
                    break

        unique_rules = []
        for r in matched_rules:
            if r not in unique_rules:
                unique_rules.append(r)
            if len(unique_rules) >= top_k:
                break

        formatted_list = []
        for idx, r in enumerate(unique_rules, 1):
            cat_name = r.get("category", "")
            title_name = r.get("title", "")
            content_str = r.get("content", "").strip()
            formatted_list.append(f"• 规范{idx}【{cat_name} - {title_name}】:\n{content_str}")

        if formatted_list:
            return "【参考病变规范与标准模板】:\n" + "\n\n".join(formatted_list)
        return ""

template_process = {
    '慢性浅表性胃炎': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以红为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C1': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C2': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，下部小弯侧黏膜红白相间，以白为主，血管透见，蠕动正常。",
        "胃角": "黏膜变薄，血管纹理显露。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C3': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，小弯黏膜地图样发红，色调逆转，延及近贲门。",
        "胃角": "黏膜变薄，血管纹理显露。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O1': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，贲门下及小弯侧黏膜菲薄，红白相间，以白为主，血管透见。",
        "胃角": "黏膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O2': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜菲薄，黏膜下血管透见，黏液湖清。",
        "胃体": "皱襞走向规则，小弯黏膜地图样发红，色调逆转，延及近贲门。",
        "胃角": "黏膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O3': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜菲薄，黏膜下血管透见，黏液湖清。",
        "胃体": "黏膜菲薄，黏膜下血管透见。",
        "胃角": "黏膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
}


# 胃镜诊断结论去重规则表
DUP_LIST = [
    [['萎缩性胃炎O3'], ['萎缩性胃炎O2'], ['萎缩性胃炎O1'], ['萎缩性胃炎C3'], ['萎缩性胃炎C2'], ['萎缩性胃炎C1'], ['萎缩性胃炎'], ['慢性浅表性胃炎'], ['浅表性胃炎']],
    [['HP现症感染'], ['Hp现症感染', 'hp现症感染', '现症感染']],
    [['贲门黏膜病变', '胃体黏膜病变', '胃角黏膜病变', '胃窦黏膜病变', '幽门黏膜病变', '十二指肠黏膜病变', '胃底黏膜病变'], ['胃黏膜病变'], ['黏膜病变'], ['胃癌']],
    [['食管黏膜病变'], ['黏膜病变'], ['食管癌']],
    [['十二指肠黏膜病变'], ['黏膜病变'], ['十二指肠癌']],
    [['黄色瘤'], ['食管黄色瘤', '胃黄色瘤']],
    [['食管乳头状瘤'], ['食管黏膜隆起'], ['食管息肉']],
    [['静脉瘤', '平滑肌瘤'], ['食管SMT'], ['食管黏膜下肿瘤'], ['食管黏膜下隆起'], ['食管黏膜隆起']],
    [['间质瘤', '异位胰腺'], ['胃SMT'], ['胃黏膜下肿瘤'], ['胃黏膜下隆起']],
    [['增生性息肉', '胃底腺息肉'], ['胃息肉'], ['胃黏膜隆起']],
    [['食管静脉曲张', '胃静脉曲张']],
    [['复合溃疡', '对吻溃疡', '霜斑样溃疡'], ['胃溃疡', '十二指肠溃疡']],
    [['斑驳食管', '反流性食管炎', '霉菌性食管炎'], ['慢性食管炎'], ['食管炎']],
    [['霉菌性食管炎'], ['真菌性食管炎']],
    [['疣状隆起'], ['疣状糜烂']],
    [['Barrett食管', '食管裂孔疝'], ['贲门松弛']],
    [['Barrett食管'], ['巴雷特食管']],
    [['贲门炎'], ['贲门糜烂']],
    [['十二指肠炎'], ['十二指肠糜烂']],
    [['食管胃黏膜异位'], ['胃黏膜异位']],
    [['肠化'], ['肠上皮化生']],
]

# 胃镜诊断结论关键词允许列表 (自动提纯 DUP_LIST 中出现的所有疾病名称，并补充 '门脉高压性胃病'、'食管溃疡')
GASTRO_ALLOWED_KEYWORDS = list({
    item for group in DUP_LIST for tier in group for item in tier
} | {'门脉高压性胃病', '食管溃疡'})

def normalize_disease_name(name: str) -> str:
    """归一化诊断疾病名称，去除‘型’字或做形式统一（如 萎缩性胃炎C2型 -> 萎缩性胃炎C2）"""
    name = name.strip()
    m = re.search(r'([COco])\s*([123])', name)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        if any(keyword in name for keyword in ['萎缩', 'atrophy', 'c', 'o', 'C', 'O']):
            return f"萎缩性胃炎{prefix}{num}"
            
    if '慢性浅表' in name or '浅表性' in name:
        return "慢性浅表性胃炎"
    if '萎缩性胃炎' in name or '慢性萎缩性胃炎' in name:
        return "萎缩性胃炎"
        
    return name


def deduplicate_concl(conclusion: list) -> tuple:
    """去重结论性疾病 (基于 DUP_LIST 优先级组)"""
    conclusion = [normalize_disease_name(item) for item in conclusion]
    word_to_tier = {}
    for group_idx, group in enumerate(DUP_LIST):
        for tier_idx, tier in enumerate(group):
            for item in tier:
                word_to_tier[item] = (group_idx, tier_idx)
                
    group_highest_triggered = {}
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            if group_idx not in group_highest_triggered:
                group_highest_triggered[group_idx] = tier_idx
            else:
                group_highest_triggered[group_idx] = min(group_highest_triggered[group_idx], tier_idx)
                
    result = []
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            if tier_idx == group_highest_triggered[group_idx]:
                result.append(item)
        else:
            result.append(item)
            
    return list(dict.fromkeys(result)), group_highest_triggered


def extract_21_diseases(text: str) -> list:
    """从 final_report 的 <think> 过程中提取 2.1 结论性疾病列表"""
    diseases = []
    # 正则精准匹配步骤编号 2.1（例如 "2.1"、"2.1："、"2.1."、"2.1、"）避免误切 "2.1cm" 等尺寸
    m21 = re.search(r'(?:^|\n)\s*2\.1[\s:：、.]', text)
    if m21:
        p21_part = text[m21.end():]
        m22 = re.search(r'(?:^|\n)\s*2\.2[\s:：、.]', p21_part)
        if m22:
            p21_part = p21_part[:m22.start()]
        if '：' in p21_part:
            p21_part = p21_part.split('：', 1)[1]
        elif ':' in p21_part:
            p21_part = p21_part.split(':', 1)[1]
        p21_part = p21_part.strip()
        lines = [line.strip() for line in p21_part.splitlines() if line.strip()]
        for line in lines:
            if line in ['无', '无。', '无', 'None']:
                continue
            items = re.split(r'[；;,、\n]', line)
            for item in items:
                item_clean = item.strip()
                if item_clean and item_clean not in ['无', '无。', 'None']:
                    diseases.append(item_clean)
    return diseases


GASTRO_PARTS = [
    "食管", "贲门", "胃底", "胃体", "胃角", "胃窦", "幽门", "十二指肠"
]


def resolve_cancer_site(first_elem: str, match_text: str) -> str:
    """根据严格约束判定癌症结论的部位 + '黏膜病变'：
    1. 部位只能是 GASTRO_PARTS 中的一个；
    2. 若 first_elem 在 GASTRO_PARTS 中，取 first_elem + '黏膜病变'；
    3. 若都不满足，看哪些 GASTRO_PARTS 存在于 match_text 语句中：
       - 若恰好 1 个，使用该 part + '黏膜病变'；
       - 若存在多个，返回 '黏膜病变'；
       - 若都不存在，检查语句中是否存在 '胃'：
         - 若存在 '胃'，返回 '胃黏膜病变'；
         - 若不存在 '胃'，返回 '黏膜病变'。
    """
    first_elem_clean = first_elem.strip()
    if first_elem_clean in GASTRO_PARTS:
        return f"{first_elem_clean}黏膜病变"

    matched_parts = [part for part in GASTRO_PARTS if part in match_text]

    if len(matched_parts) == 1:
        return f"{matched_parts[0]}黏膜病变"
    elif len(matched_parts) > 1:
        return "黏膜病变"
    else:
        if "胃" in match_text:
            return "胃黏膜病变"
        else:
            return "黏膜病变"

# todo： 后续优化项，在该函数中，默认2.2是len(list)==3,且满足（部位, 描述, 结论）的逻辑，但在实际think中，可能会缺乏部位/结论，因此其校准结论就存在描述。目前暂且通过GASTRO_ALLOWED_KEYWORDS进行限制。
def extract_22_diseases(text: str) -> list:
    """从 final_report 的 <think> 过程中提取 2.2 每个序列 list 的最后一个元素（病变名称）
    若包含 '食管癌'、'胃癌'、'十二指肠癌'，则转换成 '部位' + '黏膜病变'
    """
    diseases = []
    cancer_keywords = ['食管癌', '胃癌', '十二指肠癌']
    # 正则精准匹配步骤编号 2.2（例如 "2.2"、"2.2："、"2.2."、"2.2、"）避免误切 "2.2cm" 等尺寸
    m22 = re.search(r'(?:^|\n)\s*2\.2[\s:：、.]', text)
    if m22:
        p22_part = text[m22.end():]
        m23 = re.search(r'(?:^|\n)\s*2\.3[\s:：、.]', p22_part)
        if m23:
            p22_part = p22_part[:m23.start()]
        bracket_matches = re.findall(r'\[([^\]]+)\]', p22_part)
        for match in bracket_matches:
            if '部位' in match and '特征描述' in match:
                continue
            parts = [p.strip() for p in re.split(r'[,]', match) if p.strip()]
            if parts:
                first_elem = parts[0]
                last_elem = parts[-1]
                if last_elem and last_elem not in ['无', '无。', '病变名称']:
                    if any(ck == last_elem or ck in last_elem for ck in cancer_keywords):
                        diseases.append(resolve_cancer_site(first_elem, match))
                    else:
                        diseases.append(last_elem)
    return diseases


def process_gastro_report(extracted_report: dict) -> dict:
    """
    胃镜报告专属后处理逻辑：
    1. 诊断结论去重与模板 Key 提取：
       - 对诊断结论列表执行 deduplicate_concl（基于 DUP_LIST 优先级去重）；
       - 过滤允许关键词 GASTRO_ALLOWED_KEYWORDS；
       - 检查是否存在 template_process 中的 key。若存在，取匹配到的 key 作为模板 key；
       - 若不存在任何 template_process key，默认采用 "慢性浅表性胃炎" 模板，并在诊断结论中追加 "慢性浅表性胃炎"。
    2. 镜检所见 8 个部位解析与模板兜底补全：
       - 检查 8 个解剖部位 GASTRO_PARTS；
       - 若哪一个部位缺少，在语句中定位出现位置，将该位置到 "。" 的内容或者下一步 "部位" 前，作为该部位的输入；
       - 若全文未找到该部位，参考 template_process 中对应模版的部位补充。
    """
    if not isinstance(extracted_report, dict):
        return extracted_report

    # 1. 诊断结论后处理
    matched_template_key = None
    if "诊断结论" in extracted_report:
        raw_concl = extracted_report["诊断结论"]
        if isinstance(raw_concl, str):
            report_concl = [c.strip() for c in re.split(r'[；;.。;\n]', raw_concl) if c.strip()]
        elif isinstance(raw_concl, list):
            report_concl = [str(c).strip() for c in raw_concl if str(c).strip()]
        else:
            report_concl = []

        if report_concl:
            # 执行 DUP_LIST 去重
            dedup_report_concl, _ = deduplicate_concl(report_concl)
            dedup_report_concl = [item for item in dedup_report_concl if any(kw in item for kw in GASTRO_ALLOWED_KEYWORDS)]
        else:
            dedup_report_concl = []

        # 判断是否存在 template_process 中的 key
        for item in dedup_report_concl:
            norm_item = normalize_disease_name(item)
            if norm_item in template_process:
                matched_template_key = norm_item
                break
            elif item in template_process:
                matched_template_key = item
                break

        # 如果不存在 template_process 的 key 值，默认采用“慢性浅表性胃炎”，并在诊断结论中加入“慢性浅表性胃炎”
        if not matched_template_key:
            matched_template_key = "慢性浅表性胃炎"
            if "慢性浅表性胃炎" not in dedup_report_concl:
                dedup_report_concl.append("慢性浅表性胃炎")

        extracted_report["诊断结论"] = "；".join(dedup_report_concl)
    else:
        matched_template_key = "慢性浅表性胃炎"
        extracted_report["诊断结论"] = "慢性浅表性胃炎"

    # 2. 镜检所见后处理
    if "镜检所见" in extracted_report:
        findings = extracted_report["镜检所见"]
        parsed_findings = {}

        if isinstance(findings, dict):
            lines = [f"{k}：{v}" for k, v in findings.items()]
            raw_text = "\n".join(lines)
        elif isinstance(findings, str):
            lines = findings.splitlines()
            raw_text = findings
        else:
            raw_text = str(findings)
            lines = raw_text.splitlines()

        # Step 1: 先将 extracted_report["镜检所见"] 按照 '\n' 换行，解析正常的 "部位：描述"
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if "：" in line_str:
                k, v = line_str.split("：", 1)
                k_clean = k.strip()
                if k_clean in GASTRO_PARTS:
                    parsed_findings[k_clean] = v.strip()
            elif ":" in line_str:
                k, v = line_str.split(":", 1)
                k_clean = k.strip()
                if k_clean in GASTRO_PARTS:
                    parsed_findings[k_clean] = v.strip()

        # Step 2: 查看哪个部位不存在，若部位不存在，在全文中定位该“部位”；截取从该部位到下一个“\n”出现前的内容
        tpl_dict = template_process.get(matched_template_key, template_process["慢性浅表性胃炎"])
        final_findings = {}

        for part in GASTRO_PARTS:
            if part in parsed_findings and parsed_findings[part]:
                final_findings[part] = parsed_findings[part]
            else:
                # 在全文中寻找缺失部位的出现位置
                pos = raw_text.find(part)
                if pos != -1:
                    search_sub = raw_text[pos:]

                    # 寻找在当前部位之后出现的下一个 "\n" 位置
                    newline_idx = search_sub.find("\n")
                    if newline_idx != -1:
                        extracted_str = search_sub[:newline_idx].strip()
                    else:
                        extracted_str = search_sub.strip()

                    # 去除部位前缀及冒号
                    if extracted_str.startswith(part):
                        val = extracted_str[len(part):].lstrip("：: ")
                        final_findings[part] = val if val else extracted_str
                    else:
                        final_findings[part] = extracted_str
                else:
                    # 全文中未找到该部位，参考 template_process 对应模版补充
                    final_findings[part] = tpl_dict.get(part, "")

        extracted_report["镜检所见"] = final_findings

    return extracted_report


class Qwen3ASR17BGGUFEngine:
    """
    Qwen3-ASR-1.7B 原生 GGUF 多模态语音转写引擎
    """
    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int = 4096):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到 ASR LLM GGUF 模型: {model_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"未找到 ASR Audio mmproj GGUF 模型: {mmproj_path}")

        print(f"正在加载 ASR 引擎 (Qwen3-ASR LLM & mmproj)...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            seed=42,       # 显式固定 C++ 随机种子为 42
            verbose=False
        )

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
            raise RuntimeError("加载 ASR MTMD 音频编码上下文失败！")

    def transcribe(self, audio_path: str, prompt_text: str = "语音转写成中文：") -> str:
        """完全对齐 test_gguf.py 的 ASR 预处理与转写逻辑"""
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self.llm._ctx.ctx), True)
        
        # 使用与 test_gguf.py 完全相同的临时文件转码逻辑
        tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav = tf.name
        tf.close()

        try:
            clean_env = os.environ.copy()
            if "LD_LIBRARY_PATH" in clean_env:
                clean_env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + clean_env["LD_LIBRARY_PATH"]
            
            ffmpeg_cmd = ["/usr/bin/ffmpeg", "-y", "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env)

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
                token_id = llama_cpp.llama_sampler_sample(smpl, self.llm._ctx.ctx, -1)
                if token_id in [self.llm.token_eos(), 151645]:
                    break

                token_str = self.llm.detokenize([token_id]).decode('utf-8', errors='ignore')
                generated_text += token_str

                batch = llama_cpp.llama_batch_get_one(
                    (llama_cpp.llama_token * 1)(token_id),
                    1
                )
                if llama_cpp.llama_decode(self.llm._ctx.ctx, batch) != 0:
                    break

            llama_cpp.llama_sampler_free(smpl)

            mtmd.mtmd_bitmap_free(bitmap)
            mtmd.mtmd_input_chunks_free(chunks)

            # 清理格式（完全对齐 test_gguf.py）
            text = generated_text.replace("<|im_end|>", "").strip()
            if "<asr_text>" in text:
                text = text.split("<asr_text>")[-1].strip()
            return text
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


def detect_gi_type(text: str, audio_path: str = "") -> str:
    """对 ASR 识别出的文本及音频路径进行关键字智能匹配扫描，打标 gi_type 标签"""
    # 0. 优先从文件路径做基准判定
    path_lower = audio_path.lower()
    if any(k in path_lower for k in ["colon", "肠", "直肠", "结肠", "肛周", "痔疮", "lst", "ssl", "直肠炎"]):
        return "colon"
    if any(k in path_lower for k in ["gastro", "食管", "贲门", "幽门", "十二指肠", "胃癌"]):
        return "gastro"

    # 清理前缀提示词的影响
    text_clean = re.sub(r"^提取有效信息.*?[：:]\s*", "", text or "").strip()

    gastro_keywords = [
        "食管", "贲门", "幽门", "十二指肠", "巴雷特", "Barrett", 
        "胃炎", "胃底", "胃体", "胃角", "胃窦", "胃腔", "萎缩", "肠化", "平滑肌瘤"
    ]
    colon_keywords = [
        "结肠", "直肠", "盲肠", "回盲", "阑尾", "肛周", "内痔", "外痔", 
        "BBPS", "波士顿", "Boston", "降结肠", "乙状结肠", "升结肠", "横结肠",
        "直乙交界", "肠道", "粪水", "粪便", "息肉钳除"
    ]

    has_gastro = any(kw in text_clean for kw in gastro_keywords)
    has_colon = any(kw in text_clean for kw in colon_keywords)

    if has_gastro and has_colon:
        return "all"
    elif has_colon:
        return "colon"
    elif has_gastro:
        return "gastro"
    else:
        raise ValueError('未识别胃肠镜类型')


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    鲁棒解析大模型输出文本中的 JSON 字典对象（支持嵌套字典、Markdown 代码块及混杂文本）
    """
    if not text:
        return None

    # 1. 优先尝试从 ```json ... ``` 代码块中提取
    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    for block in reversed(json_blocks):
        block = block.strip()
        if block.startswith('{') and block.endswith('}'):
            try:
                res = json.loads(block)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

    # 2. 括号平衡栈扫描：提取所有完整闭合的最外层 {} 候选块
    candidates = []
    stack = []
    start_idx = -1

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    candidates.append(text[start_idx:i+1])
                    start_idx = -1

    # 3. 逆向（从后往前）优先筛选包含报告关键 key 的 JSON 对象
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and any(k in obj for k in ["镜检所见", "诊断结论", "检查过程", "检查结果"]):
                return obj
        except Exception:
            continue

    # 4. 逆向尝试任意合法 JSON 字典
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def fix_boston_scores(text: str) -> str:
    """自动化波士顿 (BBPS) 肠道准备评分总分纠错"""
    pattern = r"总评分\s*(\d+)\s*分[，,\s]*右半结肠（盲肠、升结肠）\s*(\d+)\s*分[、,\s]*横结肠\s*(\d+)\s*分[、,\s]*左半结肠（降结肠、乙状结肠、直肠）\s*(\d+)\s*分"
    
    def replace_fn(match):
        tot, r, m, l = map(int, match.groups())
        r_c = max(0, min(3, r))
        m_c = max(0, min(3, m))
        l_c = max(0, min(3, l))
        correct_tot = r_c + m_c + l_c
        return f"总评分{correct_tot}分，右半结肠（盲肠、升结肠）{r_c}分、横结肠{m_c}分、左半结肠（降结肠、乙状结肠、直肠）{l_c}分"

    return re.sub(pattern, replace_fn, text)


def process_single_audio(
    audio_path: str,
    gi_type_option: str,
    asr_engine: Qwen3ASR17BGGUFEngine,
    llm_engine: Llama,
    corrector: Optional[Any],
    output_dir: str,
    rag_retriever: Optional[MedicalRAGRetriever] = None,
    top_k: int = 4
) -> dict:
    """处理单个音频/视频文件并导出报告 JSON"""
    # 测量时长
    try:
        wav_info = sf.info(audio_path)
        audio_duration = wav_info.duration
    except Exception:
        audio_duration = 0.0

    # 3.1 ASR 识别 (完全对齐 test_gguf.py)
    asr_start_time = time.time()
    raw_asr_text = asr_engine.transcribe(audio_path)

    if corrector is not None:
        corrected_res = corrector.correct(raw_asr_text)
        final_asr_text = corrected_res.text
    else:
        final_asr_text = raw_asr_text

    asr_duration = time.time() - asr_start_time

    # 3.2 智能类型判定与 System Prompt 选择
    if gi_type_option in ["gastro", "colon"]:
        gi_type = gi_type_option
    else:
        try:
            gi_type = detect_gi_type(final_asr_text, audio_path)
        except ValueError as e:
            print(f"⚠️ {e}，默认降级为胃镜 (gastro) 处理。")
            gi_type = "gastro"

    if gi_type == "gastro":
        base_sys_prompt = "你是一个严谨的胃镜专家，请精准提取医生口语中的病变部位与特征描述，并结合标准胃镜规范生成结构化镜检所见与诊断结论，严禁漏诊与误诊。"
    elif gi_type == "colon":
        base_sys_prompt = "你是一个严谨的肠镜专家，请精准提取医生口语中的阳性病变与关键信息，并结合标准肠镜模板自动规范补充未见异常部位的阴性描述与诊断结论，严禁漏诊与误诊。"
    else:
        gi_type = "gastro"
        base_sys_prompt = "你是一个严谨的胃镜专家，请精准提取医生口语中的病变部位与特征描述，并结合标准胃镜规范生成结构化镜检所见与诊断结论，严禁漏诊与误诊。"

    prompt_prefix = "提取有效信息,生成标准胃镜报告：" if gi_type == "gastro" else "提取有效信息,生成标准肠镜报告："
    user_input_prompt = f"{prompt_prefix}{final_asr_text}"
    if gi_type == "gastro":
        user_input_prompt += "？"

    # 若启用 RAG 检索增强规范注入
    if rag_retriever is not None:
        retrieved_knowledge = rag_retriever.retrieve(user_input_prompt, sys_prompt=base_sys_prompt, top_k=top_k)
        if retrieved_knowledge:
            system_prompt = f"{base_sys_prompt}\n\n{retrieved_knowledge}"
        else:
            system_prompt = base_sys_prompt
    else:
        system_prompt = base_sys_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input_prompt}
    ]

    # 3.3 LLM 报告生成
    llm_start_time = time.time()
    completion = llm_engine.create_chat_completion(
        messages=messages,
        temperature=0.0,
        # repeat_penalty=1.1,
        max_tokens=2048
    )

    raw_report = completion["choices"][0]["message"]["content"] or ""
    llm_duration = time.time() - llm_start_time

    # 后处理 1: 波士顿评分纠错
    if gi_type == 'colon':
        report_with_boston = fix_boston_scores(raw_report)
    else:
        report_with_boston = raw_report

    # 后处理 2: 对 LLM 输出的报告使用 hotword_path 医学热词表再次进行同音字/错别字纠错
    if corrector is not None:
        corrected_report_obj = corrector.correct(report_with_boston)
        final_report = corrected_report_obj.text
    else:
        final_report = report_with_boston

    # 如果是胃镜，从 final_report 的 2.1 部分和 2.2 部分提取出校准结论
    intermediate_calibrated_concl = []
    if gi_type == "gastro":
        list_21 = extract_21_diseases(final_report)
        list_22 = extract_22_diseases(final_report)
        intermediate_calibrated_concl = list_21 + list_22

    total_duration = asr_duration + llm_duration

    # 控制台打印
    print("=" * 60)
    print(f"【输入音频文件】: {audio_path} (时长: {audio_duration:.2f}s | 检查类型标签: {gi_type})")
    print("-" * 50)
    print(f"【ASR 识别结果】 (耗时 {asr_duration:.2f}s):\n{final_asr_text}")
    print("-" * 50)
    print(f"【LLM 报告生成结果】 (耗时 {llm_duration:.2f}s):\n{final_report}")
    if gi_type == "gastro":
        print("-" * 50)
        print(f"【中间校准结论 (2.1 + 2.2 提取)】: {intermediate_calibrated_concl}")
    print("-" * 50)
    print(f"【流水线总耗时】: {total_duration:.2f} 秒")
    print("=" * 60)

    # 3.4 鲁棒提取镜检所见和诊断结论额外保存为 key "report"
    extracted_report = extract_json_from_text(final_report)

    if not isinstance(extracted_report, dict):
        extracted_report = {"镜检所见": final_report, "诊断结论": ""}

    # 胃镜报告 (gi_type == "gastro")：调用专属后处理 (诊断结论模板匹配与部位补全)
    report_raw_str = json.dumps(extracted_report, ensure_ascii=False) if isinstance(extracted_report, dict) else str(extracted_report)
    is_colon_content = any(ck in report_raw_str for ck in ["Boston", "波士顿", "结肠", "直肠", "回盲瓣", "阑尾", "肛周", "直乙交界"])

    # if gi_type == "gastro" and isinstance(extracted_report, dict) and not is_colon_content:
    if gi_type == "gastro" and isinstance(extracted_report, dict):
        if intermediate_calibrated_concl:
            extracted_report["诊断结论"] = intermediate_calibrated_concl
        extracted_report = process_gastro_report(extracted_report)

    # 3.5 保存 JSON 结果
    file_basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_json_path = os.path.join(output_dir, f"{file_basename}_report.json")
    
    result_data = {
        "audio_path": audio_path,
        "audio_duration_sec": round(audio_duration, 2),
        "gi_type_tag": gi_type,
        "asr_raw_text": raw_asr_text,
        "asr_corrected_text": final_asr_text,
        "report_output": final_report,
        "intermediate_calibrated_concl": intermediate_calibrated_concl if gi_type == "gastro" else [],
        "report": extracted_report,
        "pipeline_metrics": {
            "asr_duration_sec": round(asr_duration, 2),
            "llm_duration_sec": round(llm_duration, 2),
            "total_duration_sec": round(total_duration, 2)
        }
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"结果已成功导出至 JSON: {out_json_path}\n")
    return result_data


def main():
    parser = argparse.ArgumentParser(description="消化内镜语音识别 (ASR GGUF) + 内镜报告生成 (LLM GGUF) 端到端推理流水线")
    
    # 音频/媒体输入参数（支持文件与文件夹路径）
    parser.add_argument(
        "--audio_path",
        type=str,
        default="/media/inno/LLM/GI/TrainData/V2/audio/val/胃镜",
        # default="/media/inno/ASR/胃镜/audio/test/汇总/",
        # default="/media/inno/LLM/GI/TrainData/V2/audio/val/食管黏膜隆起/138_166.wav",
        help="输入的音频/视频文件路径或文件夹路径 (支持 wav, mp3, m4a, flac, aac, mp4 等)"
    )
    
    # ASR 模型参数
    parser.add_argument(
        "--asr_model_dir",
        type=str,
        default="/media/inno/work_dirs/ASR/qwen3-asr/qwen3-asr-1.7b-v4/gguf",
        help="Qwen3-ASR GGUF 目录"
    )
    parser.add_argument(
        "--asr_llm_name",
        type=str,
        default="Qwen3-ASR-1.7B-Q8_0.gguf",
        help="ASR LLM GGUF 文件名"
    )
    parser.add_argument(
        "--asr_mmproj_name",
        type=str,
        default="mmproj-Qwen3-ASR-1.7B-bf16.gguf",
        help="ASR Audio Projector GGUF 文件名"
    )
    parser.add_argument(
        "--hotword_path",
        type=str,
        default="/media/inno/ASR/gi_hotwords.txt",
        help="ASR 后处理医学热词纠错词表"
    )

    # LLM 模型参数
    parser.add_argument(
        "--llm_model_path",
        type=str,
        default="/media/inno/work_dirs/LLM/LlamaFactory/gi/outputs-sft-qwen3.5-4b-v2-lora-32-64-0.05-warmup-0.05-decay-0.05-batch4-lr1e-4-neftune5-packing-rslora/gguf/Qwen3.5-4B-Q8_0.gguf",
        help="GGUF 格式的 LLM 模型绝对路径"
    )

    # RAG 检索增强参数
    parser.add_argument(
        "--rag",
        action="store_true",
        help="是否启用 RAG 医疗规范检索增强 (默认不开启)"
    )
    parser.add_argument(
        "--kb_dir",
        type=str,
        default="/media/inno/LLM/RAG_knowledeg",
        help="RAG 医疗规范知识库 rules.json 所在的文件夹路径"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="RAG 检索返回的最大规范条数 (默认: 4)"
    )

    # 检查类型打标参数
    parser.add_argument(
        "--gi_type",
        type=str,
        default="gastro",
        choices=["all", "gastro", "colon"],
        help="人为指定检查类型 (all: 自动判定, gastro: 强制胃镜, colon: 强制肠镜)"
    )

    # 输出导出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/inno/output/LLM/gi/gi_report_rag/",
        help="报告 JSON 保存的文件夹路径"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 解析音频/媒体文件列表 (支持单个文件 & 文件夹扫描)
    MEDIA_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".mp4", ".mkv", ".ogg", ".wma"}
    audio_files = []

    if os.path.isfile(args.audio_path):
        audio_files.append(args.audio_path)
    elif os.path.isdir(args.audio_path):
        for root, _, files in os.walk(args.audio_path):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in MEDIA_EXTENSIONS:
                    audio_files.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(f"未找到指定的音频文件或文件夹路径: {args.audio_path}")

    if not audio_files:
        print(f"警告: 路径 {args.audio_path} 下未检索到任何可处理的媒体文件！")
        return

    print("=" * 60)
    print("========== 消化内镜端到端语音转报告 (ASR GGUF + LLM GGUF) 流水线 ==========")
    print(f"检索到待处理媒体文件共 {len(audio_files)} 个 | RAG 检索增强: {'开启 ✅' if args.rag else '关闭 ❌'}")
    print("=" * 60)

    # 1. 初始化 ASR 引擎
    print("\n[阶段 1/3] 初始化 ASR 语音识别引擎...")
    asr_load_start = time.time()
    asr_llm_path = os.path.join(args.asr_model_dir, args.asr_llm_name)
    asr_mmproj_path = os.path.join(args.asr_model_dir, args.asr_mmproj_name)
    asr_engine = Qwen3ASR17BGGUFEngine(model_path=asr_llm_path, mmproj_path=asr_mmproj_path)
    print(f"ASR 引擎加载完成，耗时: {time.time() - asr_load_start:.2f} 秒。")

    # 加载 ASR 后处理医学热词
    corrector = None
    if PhonemeCorrector is not None and os.path.exists(args.hotword_path):
        print(f"加载 ASR 后处理医学热词词表: {args.hotword_path}")
        corrector = PhonemeCorrector(threshold=0.85)
        with open(args.hotword_path, "r", encoding="utf-8") as f:
            corrector.update_hotwords(f.read())

    # 初始化 RAG 检索增强器 (若启用)
    rag_retriever = None
    if args.rag:
        print(f"\n[RAG 模块] 开启 RAG 检索增强，初始化知识库 ({args.kb_dir})...")
        rag_retriever = MedicalRAGRetriever(kb_dir=args.kb_dir)

    # 2. 初始化 LLM 报告生成引擎
    print("\n[阶段 2/3] 初始化 LLM 报告生成引擎...")
    llm_load_start = time.time()
    llm_engine = Llama(
        model_path=args.llm_model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        seed=42,       # 显式固定 C++ 随机种子为 42
        verbose=False
    )
    print(f"LLM 报告生成模型加载完成，耗时: {time.time() - llm_load_start:.2f} 秒。\n")

    # 3. 批量处理输入语音并生成报告
    print(f"[阶段 3/3] 开始批量处理 {len(audio_files)} 个语音/媒体文件并生成结构化报告...\n")
    summary_results = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"---- [正在处理 {idx}/{len(audio_files)}] {os.path.basename(audio_file)} ----")
        res = process_single_audio(
            audio_path=audio_file,
            gi_type_option=args.gi_type,
            asr_engine=asr_engine,
            llm_engine=llm_engine,
            corrector=corrector,
            output_dir=args.output_dir,
            rag_retriever=rag_retriever,
            top_k=args.top_k
        )
        summary_results.append(res)

    print("=" * 60)
    print(f"🎉 全部 {len(audio_files)} 个媒体文件处理完成！报告 JSON 已保存至: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
