"""Shared experiment orchestration helpers, 12-row ablation matrix, counterfactuals, and symmetric stress perturber."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from collections import defaultdict
from pypinyin import Style, pinyin
from mvcsn.data.augmentation import GLYPH_MAP, PHON_MAP, SPLIT_MAP
from mvcsn.data.preprocessing import is_cjk
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord
from mvcsn.data.vocab import CharVocab

DEFAULT_SEEDS: Tuple[int, ...] = (42, 1024, 2025)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_phonetic: bool = True
    use_glyph: bool = True
    use_gate: bool = True
    use_multimodal_anchor: bool = True
    use_consistency: bool = True
    use_oov_bridge: bool = True
    use_preprocessor: bool = True
    use_pretrained_init: bool = True
    use_domain_kd: bool = True
    use_unlabeled_kd: bool = True
    use_cross_view_kd: bool = True
    use_space_aug: bool = True


ABLATION_SPECS: Tuple[AblationSpec, ...] = (
    AblationSpec("Full MV-CSN v3 (Dual-Teacher KD)"),
    AblationSpec("w/o Phonetic View", use_phonetic=False),
    AblationSpec("w/o Glyph Visual View", use_glyph=False),
    AblationSpec("w/o Context Gate (Uniform)", use_gate=False),
    AblationSpec("w/o RoCBert Consistency", use_consistency=False),
    AblationSpec("w/o Multimodal Anchor", use_multimodal_anchor=False),
    AblationSpec("w/o Dynamic OOV Bridge", use_oov_bridge=False),
    AblationSpec("w/o Split/Noise Preprocessor", use_preprocessor=False),
    AblationSpec("w/o Pretrained Char Init", use_pretrained_init=False),
    AblationSpec("w/o Domain Teacher KD (Logit+Feat)", use_domain_kd=False),
    AblationSpec("w/o Unlabeled Transfer KD", use_unlabeled_kd=False),
    AblationSpec("w/o Cross-View & Space Aug", use_cross_view_kd=False, use_space_aug=False),
)


COUNTERFACTUAL_16_ORIG: Tuple[str, ...] = (
    "妈，我今晚加班，不用等我吃饭了。",
    "明天上午十点开会，请大家准时到会议室！",
    "你好，你的快递已放丰巢柜，请及时取件。",
    "周末有空吗，一起去看电影吧！",
    "收到，谢谢老师，我马上修改完发您邮箱。",
    "今天天气真好，我们去操场跑步吧。",
    "提醒：明天截止提交课程作业，请记得上传系统。",
    "您好，您预约的明天下午2点门诊已确认，请携带身份证前往。",
    "晚上一起去食堂吃晚饭吧？我六点半下课。",
    "实验报告我已经写完了，等会儿发到群里大家核对一下。",
    "今天下雨记得带伞，路上注意安全！",
    "恭喜你通过初试，请下周三上午带简历来面试。",
    "火车站人好多啊，我提前半小时到检票口等你。",
    "老师好，请问这周的数据工程课是在东校园上吗？",
    "昨天买的书已经发货了，预计后天上午送到宿舍楼下。",
    "我手机快没电了，回去之后再给你回电话。",
)


COUNTERFACTUAL_40_BALANCED: Tuple[Tuple[int, str], ...] = (
    # 20 everyday Ham (16 orig + 4 transactional service notices)
    *[(0, t) for t in COUNTERFACTUAL_16_ORIG],
    (0, "【中国银行】您尾号8421的借记卡于今日10:15支取人民币120.00元，当前余额3580.40元。"),
    (0, "【顺丰速运】您的包裹已由菜鸟驿站代收，取件码5-2-308，请于晚9点前领取。"),
    (0, "【教务处】各位同学，本周五下午两点的学术讲座改至公教楼B201举行，请互相转告。"),
    (0, "您的登录动态验证码为639284，5分钟内有效，请勿向他人泄露。"),
    # 20 realistic diverse Spam (matching punctuation & notification disguises)
    (1, "尊敬的客户，您已获得10万元无抵押低息备用金，点击链接即可秒批到账！"),
    (1, "特惠充100送200，注册即送888元彩金，加微信号vip8899领取专属福利。"),
    (1, "本公司长期代开各类正规增值税发票，点数低可验后付款，详询电话13800138000。"),
    (1, "【系统通知】您的积分将于今晚清零，请立即访问http://jf-dh99.cn兑换现金大礼包！"),
    (1, "居家兼职日结300到800元，只需手机点赞刷单，加QQ群77889900立刻上岗。"),
    (1, "恭喜您成为本周幸运用户，免费领取最新款手机一部，请速联系客服登记邮寄地址！"),
    (1, "【紧急通知】您有一笔法院传票未签收，请于今日17点前拨打010-88776655核实身份。"),
    (1, "澳门新葡京官方直营，首存翻倍返水高达3%，全天候提现秒到，网址8899vip.com。"),
    (1, "你好，我是房产中介小李，市中心地铁口精装公寓低价急售，周末看房享9折优惠！"),
    (1, "专业办理高额度信用卡套现及代还业务，手续费低至千分之五，有需要请回电。"),
    (1, "内部消息明天必涨停牛股三只，免费领股加微信niugu666，名额仅限前50名！"),
    (1, "本店周年庆全场大促，扫码加好友立享198元美容套餐，先到先得回T退订。"),
    (1, "您的ETC卡片状态异常已暂停使用，请点击etc-update.cc重新认证以免影响高速通行。"),
    (1, "正规学历提升免试入学，一年拿本科毕业证学信网可查，咨询热线400-888-9999。"),
    (1, "诚邀参加高端理财私享会，年化收益保底18%稳赚不赔，回复1立即预留席位！"),
    (1, "【银行提醒】您的账户存在异常交易已被冻结，请尽快登录icbc-safe88.cn解冻。"),
    (1, "最新棋牌麻将智能控牌工具，稳赢不输包教包会，有意者加微majiang998详谈。"),
    (1, "全国各大城市均可安排上门高端保健按摩服务，照片本人可挑，联系微信spa6688。"),
    (1, "恭喜您获批20万信用贷款额度，免担保不查征信，点击t.cn/dk8899立即提现。"),
    (1, "特价清仓品牌名表一比一复刻，货到付款支持开箱验货，加微biao777看款下单。"),
)


def to_clean_record(
    record: SpamRecord,
    preproc: TextPreprocessor,
    max_len: int = 192,
) -> SpamRecord:
    clean_raw = record.clean_text if record.clean_text else record.raw_text
    clean_proc = preproc.process(clean_raw, enabled=True) or clean_raw
    return SpamRecord(
        idx=record.idx,
        label=record.label,
        raw_text=clean_raw,
        proc_text=clean_proc,
        clean_text=clean_raw,
        cluster_id=record.cluster_id,
        split=record.split,
        attack_type="CLEAN",
        obf_mask=[0] * min(len(clean_proc), max_len),
    )


def build_symmetric_stress_perturber(
    vocab: CharVocab,
    chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
) -> Callable[[str, float, random.Random, str], str]:
    """Build full-vocab symmetric perturbation engine perturbing BOTH Ham & Spam at equal rates."""
    pre = TextPreprocessor(chaizi_path)
    py_groups: Dict[str, List[str]] = defaultdict(list)
    for ch in vocab.itos[2:]:
        if is_cjk(ch):
            py_res = pinyin(ch, style=Style.NORMAL, heteronym=False, errors="ignore")
            if py_res and py_res[0] and py_res[0][0]:
                py_groups[py_res[0][0]].append(ch)
    char_homophones: Dict[str, List[str]] = {}
    for _, group in py_groups.items():
        if len(group) >= 2:
            for ch in group:
                alts = [c for c in group if c != ch]
                if alts:
                    char_homophones[ch] = alts
    for ch, alts in PHON_MAP.items():
        merged = list(dict.fromkeys(list(alts) + char_homophones.get(ch, [])))
        char_homophones[ch] = merged

    def _perturb(text: str, rate: float, rng: random.Random, mode: str = "mixed") -> str:
        if rate <= 0.0 or not text:
            return text
        modes = ("PHON", "GLYPH") if mode == "sub_only" else ("PHON", "GLYPH", "SPLIT", "SPACE")
        out: List[str] = []
        for ch in text:
            if not (is_cjk(ch) or ch.isalnum()) or rng.random() >= rate:
                out.append(ch)
                continue
            m = rng.choice(modes)
            if m == "GLYPH" and ch in GLYPH_MAP:
                out.append(rng.choice(GLYPH_MAP[ch]))
            elif m == "SPLIT" and (ch in SPLIT_MAP or ch in pre.char_to_radicals):
                if ch in SPLIT_MAP:
                    out.append(SPLIT_MAP[ch])
                else:
                    out.append("".join(pre.char_to_radicals[ch][:2]))
            elif m == "SPACE":
                noise = rng.choice([" ", "  ", "（", "*"])
                out.append(f"（{ch}）" if noise == "（" else f"{ch}{noise}")
            elif ch in char_homophones:
                out.append(rng.choice(char_homophones[ch]))
            elif ch in GLYPH_MAP:
                out.append(rng.choice(GLYPH_MAP[ch]))
            else:
                out.append(ch)
        return "".join(out)

    return _perturb


def perturb_records_symmetrically(
    clean_records: Sequence[SpamRecord],
    perturb_fn: Callable[[str, float, random.Random, str], str],
    preproc: TextPreprocessor,
    rate: float,
    seed: int = 42,
    mode: str = "mixed",
    max_len: int = 192,
) -> List[SpamRecord]:
    """Perturb BOTH Ham and Spam symmetrically at identical rate so noise presence is never a label shortcut."""
    rng = random.Random(seed)
    out: List[SpamRecord] = []
    for rec in clean_records:
        base = rec.clean_text if rec.clean_text else rec.raw_text
        raw_p = perturb_fn(base, rate, rng, mode) if rate > 0.0 else base
        proc_p = preproc.process(raw_p, enabled=True) or raw_p
        out.append(
            SpamRecord(
                idx=rec.idx,
                label=rec.label,
                raw_text=raw_p,
                proc_text=proc_p,
                clean_text=base,
                cluster_id=rec.cluster_id,
                split=rec.split,
                attack_type=f"SYM_{mode.upper()}_{int(round(rate * 100))}",
                obf_mask=[0] * min(len(proc_p), max_len),
            )
        )
    return out


def measure_fn_ms(
    fn: Callable[[], Any],
    n_samples: int,
    warmup: int = 2,
    repeats: int = 5,
    cuda_sync: bool = True,
) -> float:
    """Measure synchronized wall-clock latency in milliseconds per sample."""
    use_sync = cuda_sync and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_sync:
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if use_sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0 / max(1, n_samples))
    return float(np.median(times))


def aggregate_metric_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        return {}
    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    return {
        key: {
            "mean": float(np.mean([float(r[key]) for r in runs])),
            "std": float(np.std([float(r[key]) for r in runs])),
        }
        for key in numeric_keys
    }


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return out


agg_stats = aggregate_metric_runs
