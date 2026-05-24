#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：1_applicant_classification.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : Classify CNIPA patent applicants into four types: individual, company, university, and government.
'''


import csv
import re
from pathlib import Path


# ============================================================
# 0. File paths
# ============================================================

INPUT_CSV = Path("path/to/applicant_cleaned.csv")
OUTPUT_CSV = Path("path/to/applicant_classified.csv")
ENCODING = "utf-8"


# ============================================================
# 1. Keyword definitions
# ============================================================

LONGNAME_KEYWORDS = [
    "中华人民共和国", "国", "省", "市", "区", "县", "镇", "村",
    "事务所", "合作社", "基金会", "出版社", "办事处", "供销社", "代表处", "事务厅",
    "图书馆", "试验所", "检验所", "工作室", "企事业", "档案馆", "服务部", "委员会",
    "董事会", "博物馆", "幼儿园", "供应商", "财团法人", "检测中心", "试验中心", "解决方案",
    "国防", "发动机", "无人机", "机器人", "半导体", "自动化",
    "卫生", "科学", "安全", "文化", "创新", "加拿大", "健康",
    "空客直升机", "GN瑞声达A*S", "卡特彼勒SARL", "阿尔卡特朗讯", "韩华思路信",
    "赛诺菲·安万特", "现代岱摩斯", "史陶比尔法万举", "比克·维尔莱克", "巴黎欧莱雅",
]

SHORTNAME_KEYWORDS = [
    "大学", "学院", "医院", "学校", "公司", "集团", "会社", "公社", "企业", "工厂", "机构",
    "研究", "有限", "合伙", "控股", "股份", "法人",
    "网络", "电器", "丹麦", "技术", "工程", "生物", "农业", "管理",
    "化学", "工业", "机械", "汽车", "动力", "材料", "化工", "电气", "电子",
    "科技", "资本", "投资", "咨询", "航空", "食品", "精密", "能源", "智能",
    "软件", "金属", "钢铁", "照明", "广告", "线材", "移动", "理工", "通信",
    "株式", "国际", "时代", "治疗", "医疗", "设计", "制造", "医药", "家私",
    "通讯", "生命", "环境", "贸易", "器械", "教育", "数据", "大臣", "部长",
    "欧洲", "联合", "家族", "精机", "仪器", "医学", "建设", "基础", "精工",
    "制品", "纳米", "药物", "特别", "密封", "外交", "环保", "实验", "基因",
    "加德士", "赛丽康",
]

SUFFIX_KEYWORDS = [
    "院", "所", "校", "局", "部", "厅", "行", "委", "司", "队", "厂", "段", "社",
    "业", "店", "坊", "庄", "城", "铺", "苑", "斋", "库", "屋", "圃", "寺", "台",
    "站", "团", "室", "场", "矿", "国", "会", "馆", "园", "处", "旅", "营", "区", "署",
    "中心", "大学", "公司", "集团", "基地", "商店", "超市", "分院", "农场", "银行", "党校",
    "商会", "大队", "单位", "协会", "医院", "小学", "中学", "机构", "政府", "技术", "代表",
    "校区", "系统", "组织", "企业", "服务", "高中", "监狱", "科技", "联盟", "合伙", "电子",
    "制药", "医疗", "智行", "洋行", "商行", "电厂", "基金", "苗圃",
]

LOCATION_PREFIXES = [
    "国家", "中国", "中央", "上海", "北京", "广州", "台湾", "香港", "新疆", "广西", "广东",
    "日本", "法国", "美国", "德国", "英国", "新加坡", "澳大利亚", "韩国", "瑞士", "瑞典",
    "意大利", "加拿大", "印度", "马来西亚", "泰国", "越南", "印度尼西亚", "菲律宾", "柬埔寨",
]

GOV_PATTERNS = [
    "市场监督", "市场检验", "市场监管",
    "检验检疫局", "养殖研究所", "种植研究所",
    "大韩民国", "美利坚众合国", "上海市松江二中",
]

KNOWN_COMPANY_NAMES = {
    "韩国", "赛丽康", "赛诺菲", "欧莱雅", "斯奈克玛",
}


# ============================================================
# 2. Regex patterns
# ============================================================

PAT_LONG = re.compile("|".join(map(re.escape, LONGNAME_KEYWORDS)))
PAT_SHORT = re.compile("|".join(map(re.escape, SHORTNAME_KEYWORDS)))
PAT_SUFFIX = re.compile("|".join(f"{re.escape(x)}$" for x in SUFFIX_KEYWORDS))

PAT_HAS_CHINESE = re.compile(r"[\u4e00-\u9fa5]")
PAT_HAS_LATIN = re.compile(r"[a-zA-Z]")
PAT_COMPANY = re.compile(r"公司")
PAT_STOCK = re.compile(r"[（(]株[）)]")
PAT_GOV_OVERRIDE = re.compile("|".join(map(re.escape, GOV_PATTERNS)))

PAT_UNI_INCLUDE = re.compile(r"(大学|学院|学校)")
PAT_UNI_EXCLUDE = re.compile(
    r"(科学院|医院|公司|厂|儿童|义务教育|进修学院|进修学校|小学|中学|干部学校|干部学院|盲人学校|实验学校|中心学校)"
)
PAT_UNI_START = re.compile(r"^(大学|学院|学校)")
PAT_UNI_END = re.compile(r"(大学|学院|学校)$")


# ============================================================
# 3. Utility functions
# ============================================================

def safe_str(value) -> str:
    """Convert missing values to an empty string."""
    if value is None:
        return ""
    return str(value)


def is_university(name: str) -> bool:
    """Classify university-like applicants."""
    whitelist = {"中国科学院大学"}

    if name in whitelist:
        return True

    if not PAT_UNI_INCLUDE.search(name):
        return False

    if PAT_UNI_EXCLUDE.search(name):
        return False

    if PAT_UNI_START.match(name):
        return bool(PAT_UNI_END.search(name))

    return True


def is_government(name: str) -> bool:
    """Classify government-like applicants."""
    if name.endswith("业") or name.endswith("店"):
        return False

    include_keywords = [
        "国家", "局", "学",
        "院", "所", "校", "局", "部", "厅", "委", "司", "队",
        "库", "寺", "站", "团", "室", "会", "段", "社",
        "馆", "园", "处", "旅", "营", "台", "区", "署",
        "机构", "单位", "代表", "系统", "组织", "高中", "监狱", "联盟",
        "政府", "研究院", "基金会", "办事处", "中心", "大臣", "部长",
        "代表处", "事务厅", "图书馆", "试验所", "检验所", "档案馆", "服务部",
        "委员会", "博物馆", "幼儿园", "中华人民共和国",
    ]

    exclude_keywords = [
        "公司", "企业", "厂", "场", "商行", "银行",
        "集团", "会社", "公社", "有限", "合伙", "控股", "股份", "林场", "种植", "养殖",
        "餐厅", "庄园", "宾馆", "财团", "事务所", "合作社", "出版社", "供销社", "工作室",
        "农业园", "销售部", "经营部", "营业部", "经销部", "经销处",
        "家庭农场", "投资中心", "咨询中心", "贸易中心",
    ]

    has_include = any(keyword in name for keyword in include_keywords)
    has_exclude = any(keyword in name for keyword in exclude_keywords)

    return has_include and not has_exclude


def classify_name(name: str) -> str:
    """Classify one applicant name."""
    name_len = len(name)

    mask_long = bool(PAT_LONG.search(name)) and name_len > 4
    mask_short = bool(PAT_SHORT.search(name)) and name_len > 3
    mask_suffix = bool(PAT_SUFFIX.search(name)) and name_len > 3

    applicant_type = "company" if mask_long or mask_short or mask_suffix else "individual"

    if not PAT_HAS_CHINESE.search(name):
        applicant_type = "company"

    if PAT_HAS_LATIN.search(name) and PAT_HAS_CHINESE.search(name) and "·" not in name:
        applicant_type = "company"

    if name_len > 4 and any(name.startswith(loc) for loc in LOCATION_PREFIXES):
        applicant_type = "company"

    if PAT_COMPANY.search(name):
        applicant_type = "company"

    if name_len == 4 and name.endswith("司") and not name.endswith("公司"):
        applicant_type = "individual"

    if name_len == 4 and name.endswith("行") and not name.endswith(("商行", "洋行", "智行")):
        applicant_type = "individual"

    if name in KNOWN_COMPANY_NAMES:
        applicant_type = "company"

    if applicant_type != "individual" and is_university(name):
        applicant_type = "university"

    if applicant_type == "company" and is_government(name):
        applicant_type = "government"

    if PAT_GOV_OVERRIDE.search(name):
        applicant_type = "government"

    if PAT_STOCK.search(name):
        applicant_type = "company"

    return applicant_type


# ============================================================
# 4. Main pipeline
# ============================================================

def main() -> None:
    """Classify applicants and save the final file."""
    name_type_cache = {}
    seen_keys = set()

    with open(INPUT_CSV, "r", encoding=ENCODING, newline="") as fin:
        reader = csv.DictReader(fin)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        fieldnames = list(reader.fieldnames)
        required_columns = ["ida", "applicant_seq", "applicant"]

        for col in required_columns:
            if col not in fieldnames:
                raise ValueError(f"Required column not found: {col}")

        output_fieldnames = list(fieldnames)

        if "type" not in output_fieldnames:
            output_fieldnames.append("type")

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_CSV, "w", encoding=ENCODING, newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                ida = safe_str(row.get("ida"))
                applicant_seq = safe_str(row.get("applicant_seq"))
                applicant = safe_str(row.get("applicant"))

                dedup_key = (ida, applicant_seq)

                if dedup_key in seen_keys:
                    continue

                seen_keys.add(dedup_key)

                applicant_type = name_type_cache.get(applicant)

                if applicant_type is None:
                    applicant_type = classify_name(applicant)
                    name_type_cache[applicant] = applicant_type

                row["type"] = applicant_type
                writer.writerow(row)


if __name__ == "__main__":
    main()

