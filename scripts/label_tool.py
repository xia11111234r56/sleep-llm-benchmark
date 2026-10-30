"""人工标注界面（Streamlit）。

启动：
    streamlit run scripts/label_tool.py

两个标注者各自启动一次，在侧边栏选自己的代号（A / B）。
两人写不同的文件，**互相看不到对方的标注** —— 这是算 Cohen's Kappa 的前提。

界面刻意做到的几件事：

1. **先倾向、后归因**。顺序固定，避免先看归因带来的顺序效应（标注手册 §5.2）。
2. **evidence 实时校验是不是原文连续子串**。不合法就当场报错，
   而不是等评测时才发现人工标注的证据没法做基线对照。
3. **预标注只作提示，可一键采纳也可完全无视**。判定规则以标注手册为准。
4. 每条保存即落盘，中断不丢。
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ATTR_LABELS = ["焦虑型", "压力型", "生理型", "环境型", "其他"]
TEND_LABELS = ["有抑郁倾向", "无抑郁倾向", "无法判断"]
ATTR_HELP = {
    "焦虑型": "紧张/担忧/心慌，且无明确外部诱因（排除法兜底，不是『看起来最像』）",
    "压力型": "由具体外部事件引发：考试、工作、人际冲突、分手、经济压力",
    "生理型": "身体因素：疼痛、服药、咖啡因、倒班、作息紊乱、更年期、孕期",
    "环境型": "环境因素：噪音、室友打呼、温度、光线、床、出差换床",
    "其他": "不属于以上四类，或信息不足（混合无主次、纯求助无背景）",
}
TEND_HELP = {
    "有抑郁倾向": "命中五类标记任一：无望感/自责/兴趣丧失/自杀意念/躯体化(不含失眠)",
    "无抑郁倾向": "五类标记均未出现",
    "无法判断": "正文不足 30 字、只有标题、或纯提问无自我状态描述",
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def item_id(r: dict) -> str:
    return r["id"]


# ---------- 数据加载 ----------
st.set_page_config(page_title="失眠文本标注", layout="wide")

# ---------- 数据集选择 ----------
DATASETS = {
    "测试集（双人独立，算 Kappa）": {
        "path": "data/splits/test_pool.jsonl",
        "pre": "data/splits/test_preannot.jsonl",
        "out": "data/labeled/annot_{who}.jsonl",
    },
    "训练集（单人标，供微调）": {
        "path": "data/splits/train.jsonl",
        "pre": "data/splits/train_preannot.jsonl",
        "out": "data/labeled/train_annot_{who}.jsonl",
    },
}

with st.sidebar:
    st.header("标注设置")
    dataset_name = st.selectbox("标注哪个数据集", list(DATASETS))
    cfg = DATASETS[dataset_name]

preannot = {r["id"]: r for r in load_jsonl(ROOT / cfg["pre"])}

pool = load_jsonl(ROOT / cfg["path"])
if not pool:
    st.error(f"找不到 `{cfg['path']}`。先运行 `python scripts/make_splits.py`。")
    st.stop()

# ---------- 侧边栏 ----------
with st.sidebar:
    annotator = st.radio("你的代号", ["A", "B"], horizontal=True,
                         help="测试集需要两个不同的人各标一份；训练集单人即可")
    out_path = ROOT / cfg["out"].format(who=annotator)

    saved = {r["id"]: r for r in load_jsonl(out_path)}
    st.caption(f"输出文件：`{cfg['out'].format(who=annotator)}`")
    st.caption(f"已完成 **{len(saved)}** / {len(pool)}")

    st.divider()
    if st.button("导出 CSV（给离线标注的人用）", use_container_width=True):
        exp = ROOT / f"data/labeled/标注模板_{dataset_name[:3]}_{annotator}.csv"
        exp.parent.mkdir(parents=True, exist_ok=True)
        with exp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "attribution", "tendency", "evidence", "note", "正文"])
            for r in pool:
                w.writerow(["", "", "", "", "", r["text"]])
        st.success(f"已导出 {exp}")
        st.caption("填好后另存为 UTF-8，再放回 data/labeled/")

    st.divider()
    st.caption("**标注手册要点**")
    st.caption("· 归因：帖主自陈因果优先 → 否则 压力>生理>环境>焦虑 → 并列无主次=其他")
    st.caption("· 倾向：指不出具体是哪类标记的哪一句，就不能判『有抑郁倾向』")
    st.caption("· evidence：原文连续子串，5–50 字，不改写")

# ---------- 定位 ----------
if "idx" not in st.session_state:
    # 从第一条未标注的开始
    first_unlabeled = next((i for i, r in enumerate(pool)
                            if item_id(r) not in saved), 0)
    st.session_state.idx = first_unlabeled

with st.sidebar:
    st.divider()
    st.number_input("跳到第几条（从 1 开始）", min_value=1, max_value=len(pool),
                    value=st.session_state.idx + 1, key="jump",
                    on_change=lambda: st.session_state.update(idx=st.session_state.jump - 1))

idx = st.session_state.idx
item = pool[idx]
iid = item_id(item)
existing = saved.get(iid, {})
suggest = preannot.get(iid, {})

# ---------- 主区域 ----------
st.subheader(f"第 {idx + 1} / {len(pool)} 条　·　`{iid}`　·　{len(item['text'])} 字")

if existing:
    st.success("这条已标注过，下面显示的是你之前的选择。改完再点保存即可覆盖。")

col_text, col_label = st.columns([3, 2])

with col_text:
    st.markdown("**原文**")
    st.text_area("原文", value=item["text"], height=380, disabled=True,
                 label_visibility="collapsed")

with col_label:
    # ---- 先倾向 ----
    st.markdown("### ① 抑郁倾向")
    t_default = TEND_LABELS.index(existing["tendency"]) if existing.get("tendency") in TEND_LABELS else None
    tendency = st.radio("倾向", TEND_LABELS, index=t_default, key=f"t_{iid}",
                        captions=[TEND_HELP[t] for t in TEND_LABELS],
                        label_visibility="collapsed")

    # ---- 后归因 ----
    st.markdown("### ② 失眠归因")
    a_default = ATTR_LABELS.index(existing["attribution"]) if existing.get("attribution") in ATTR_LABELS else None
    attribution = st.radio("归因", ATTR_LABELS, index=a_default, key=f"a_{iid}",
                           captions=[ATTR_HELP[a] for a in ATTR_LABELS],
                           label_visibility="collapsed")

    # ---- 预标注提示 ----
    if suggest.get("ok"):
        with st.expander("模型预标注建议（仅供参考，不是金标）", expanded=False):
            st.write(f"归因：**{suggest.get('suggest_attribution')}**　"
                     f"倾向：**{suggest.get('suggest_tendency')}**")
            st.caption(f"依据：{suggest.get('suggest_evidence')}")
            st.caption(f"预标注模型：{suggest.get('model')}（第三方家族，不参与实验）")

# ---- evidence ----
st.markdown("### ③ 判读依据（evidence）")
st.caption("必须是原文的**连续子串**，一字不改，5–50 字。多条用 ` | ` 分隔，最多 3 条。"
           "标『其他』且零线索、或『无法判断』时填 N/A。")

ev_default = existing.get("evidence", "") or ""
evidence = st.text_input("evidence", value=ev_default, key=f"e_{iid}",
                         placeholder="从上面原文里复制粘贴，不要改写")

ev_ok = True
if evidence.strip() and evidence.strip().upper() not in ("N/A", "NA"):
    norm_text = normalize(item["text"])
    pieces = [p.strip() for p in evidence.replace("｜", "|").split("|") if p.strip()]
    problems = []
    for p in pieces:
        if normalize(p) not in norm_text:
            problems.append(f"「{p[:30]}」不是原文连续片段")
        elif not (5 <= len(p) <= 50):
            problems.append(f"「{p[:30]}」长度 {len(p)} 字，不在 5–50 范围")
    if problems:
        ev_ok = False
        st.error("　".join(problems))
    else:
        st.success(f"✓ {len(pieces)} 个片段全部在原文中")

note = st.text_input("备注（手册未覆盖的争议点，可留空）", value=existing.get("note", ""),
                     key=f"n_{iid}")

# ---- 操作 ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("保存并下一条", type="primary", use_container_width=True):
        if attribution is None or tendency is None:
            # radio 默认不预选，不校验的话会存进 None —— 试标时实测漏过 3 条
            st.error("还没有选完整的标签：倾向和归因都要选。")
        elif not ev_ok:
            st.error("evidence 不合法，先修正再保存。")
        else:
            rec = {
                "id": iid,
                "attribution": attribution,
                "tendency": tendency,
                "evidence": evidence.strip(),
                "annotator": annotator,
                "annotated_at": date.today().isoformat(),
                "note": note.strip(),
                "text": item["text"],
            }
            saved[iid] = rec
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for r in saved.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            st.session_state.idx = min(idx + 1, len(pool) - 1)
            st.rerun()

with c2:
    if st.button("← 上一条", use_container_width=True):
        st.session_state.idx = max(idx - 1, 0)
        st.rerun()
with c3:
    if st.button("下一条 →", use_container_width=True):
        st.session_state.idx = min(idx + 1, len(pool) - 1)
        st.rerun()
with c4:
    if suggest.get("ok") and st.button("采纳预标注建议", use_container_width=True):
        st.session_state[f"t_{iid}"] = suggest["suggest_tendency"]
        st.session_state[f"a_{iid}"] = suggest["suggest_attribution"]
        st.session_state[f"e_{iid}"] = suggest.get("suggest_evidence") or ""
        st.rerun()

st.divider()
done = len(saved)
st.progress(done / len(pool), text=f"进度 {done}/{len(pool)}")
if done == len(pool):
    st.balloons()
    st.success("全部完成。把 `data/labeled/annot_*.jsonl` 交给下一步算 Kappa。")
