"""采集百度贴吧指定吧的公开帖子（主帖正文）。

重要背景 —— 为什么用浏览器而不是 requests：
    贴吧 PC 版（2026 年现状）已改为 CSR（客户端渲染），HTML 里 <div id="app"></div> 是空的，
    帖子数据由 JS 运行时填充。requests + BeautifulSoup 只能拿到空壳。
    对应的 JSON 接口 /c/f/frs/page_pc 要求登录态（匿名返回 error_code 110001）。
    实测可行的方式：有头 Edge + 持久化用户目录 + 先访问 baidu.com 建立会话。
    headless 模式会被百度安全验证识别并拦截。

伦理约束（项目方案第五部分要求，代码里强制落实）：
    - 只采集公开内容
    - 每次请求间隔 1.5 秒以上
    - 不记录用户名（避免构建用户画像）

用法：
    python crawl.py --kw 失眠 --pages 5 --limit 20      # 小批量试跑
    python crawl.py --kw 失眠 --pages 40 --out data/raw/tieba_raw.jsonl
"""

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / ".browser_profile"

LIST_URL = "https://tieba.baidu.com/f?kw={kw}&ie=utf-8&pn={pn}"
THREAD_URL = "https://tieba.baidu.com/p/{tid}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

PAGE_SIZE = 50  # 贴吧列表页每页 50 帖


def build_driver(headless: bool = False):
    opts = Options()
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--lang=zh-CN")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if headless:
        opts.add_argument("--headless=new")

    driver = webdriver.Edge(options=opts)
    # BFE 风控会检测 navigator.webdriver
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    driver.set_page_load_timeout(60)
    return driver


def is_blocked(driver) -> bool:
    return "安全验证" in driver.page_source


def sleep_a_bit(base: float):
    """请求间隔：基础间隔 + 随机抖动，避免固定节奏。"""
    time.sleep(base + random.uniform(0, 0.8))


def load(driver, url: str, wait_css: str, delay: float, retries: int = 3):
    """打开页面并等待某个元素渲染出来。失败会重试并退避。"""
    for attempt in range(retries):
        try:
            driver.get(url)
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_css))
            )
            if is_blocked(driver):
                raise RuntimeError("被百度安全验证拦截")
            return True
        except Exception as e:
            wait = delay * (2 ** attempt) + 5
            print(f"    ! 第 {attempt + 1}/{retries} 次失败（{type(e).__name__}: {e}），"
                  f"等待 {wait:.0f}s 后重试", flush=True)
            time.sleep(wait)
    return False


SCROLL_JS = "window.scrollTo(0, document.body.scrollHeight);"


def scroll_to_load_all(driver, delay: float, max_rounds: int = 12) -> int:
    """列表页是懒加载：只渲染首屏，滚动才继续加载。滚到不再增多为止。

    返回滚动加载到的帖子链接数。
    """
    last = -1
    for _ in range(max_rounds):
        driver.execute_script(SCROLL_JS)
        time.sleep(1.2)
        n = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/p/']"))
        if n <= last:
            break
        last = n
    return last


def collect_tids(driver, kw: str, pages: int, delay: float,
                 start_page: int = 0) -> list[str]:
    """翻列表页，收集帖子 tid（保持出现顺序，去重）。

    start_page 用于续采时跳过已翻过的页，省掉重复的列表开销。
    """
    seen, ordered = set(), []
    for pn in range(start_page * PAGE_SIZE, pages * PAGE_SIZE, PAGE_SIZE):
        url = LIST_URL.format(kw=quote(kw), pn=pn)
        print(f"[列表] pn={pn} ...", flush=True)
        if not load(driver, url, "a[href*='/p/']", delay):
            print(f"    x 第 {pn} 页放弃", flush=True)
            continue

        n_links = scroll_to_load_all(driver, delay)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        before = len(ordered)
        for a in soup.select("a[href*='/p/']"):
            m = re.search(r"/p/(\d+)", a.get("href", ""))
            if not m:
                continue
            tid = m.group(1)
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        print(f"    DOM 内帖子链接 {n_links} 个 -> 本页新增 {len(ordered) - before}，"
              f"累计 {len(ordered)} 个 tid", flush=True)
        sleep_a_bit(delay)
    return ordered


def extract_main_post(html: str) -> dict:
    """从帖子页 HTML 提取主帖标题与正文。

    主帖正文 = .richtext-item（唯一一个，实测可靠）；
    兜底：不在 .pb-content-item 内的 .pb-text-wrapper（回复都在 .pb-content-item 里）。
    """
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one(".pb-title")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    body = ""
    rt = soup.select_one(".richtext-item")
    if rt:
        body = rt.get_text("\n", strip=True)
    else:
        for w in soup.select(".pb-text-wrapper"):
            if w.find_parent(class_="pb-content-item") is None:
                body = w.get_text("\n", strip=True)
                break

    return {"title": title, "text": body}


def fetch_thread(driver, tid: str, delay: float) -> dict | None:
    if not load(driver, THREAD_URL.format(tid=tid), ".pb-title", delay):
        return None

    data = extract_main_post(driver.page_source)
    if not data["text"]:
        # 页面能打开但没有正文：可能是纯图片帖、已被删除、或纯标题帖
        return {"tid": tid, "title": data["title"], "text": "", "note": "no_text"}

    data["tid"] = tid
    data["url"] = THREAD_URL.format(tid=tid)
    data["text"] = re.sub(r"\n{2,}", "\n", data["text"]).strip()
    return data


def load_done(path: Path) -> set:
    """读取已采集的 tid，支持断点续采。"""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["tid"])
                    except Exception:
                        pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", default="失眠",
                    help="贴吧名，多个用逗号分隔，如 失眠,睡眠障碍,失眠症")
    ap.add_argument("--pages", type=int, default=5, help="翻到第几页列表为止（每页 50 帖）")
    ap.add_argument("--start-page", type=int, default=0,
                    help="从第几页开始翻。续采时用，省掉重复翻已抓过的页")
    ap.add_argument("--limit", type=int, default=0, help="最多抓多少帖（0=不限，用于试跑）")
    ap.add_argument("--delay", type=float, default=1.5, help="请求基础间隔秒数，勿低于 1.5")
    ap.add_argument("--out", default="data/raw/tieba_raw.jsonl")
    ap.add_argument("--headless", action="store_true", help="无头模式（实测会被风控拦截）")
    args = ap.parse_args()

    if args.delay < 1.5:
        print("请求间隔不得低于 1.5 秒（伦理约束），已强制设为 1.5")
        args.delay = 1.5

    # 多个吧走同一条命令，避免 PowerShell 传中文参数时的编码问题
    kws = [k.strip() for k in args.kw.replace("，", ",").split(",") if k.strip()]
    print(f"目标贴吧：{'、'.join(kws)}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"已有 {len(done)} 条记录，将跳过")

    driver = build_driver(headless=args.headless)
    try:
        print("[预热] 访问 baidu.com / 贴吧首页 建立会话 ...")
        driver.get("https://www.baidu.com/")
        sleep_a_bit(args.delay)
        driver.get("https://tieba.baidu.com/")
        sleep_a_bit(args.delay)

        total_ok = total_empty = total_fail = 0
        for kw in kws:
            print(f"\n{'=' * 62}\n【{kw}】")
            tids = collect_tids(driver, kw, args.pages, args.delay, args.start_page)
            done = load_done(out_path)  # 每个吧重新读一次，前面吧的结果也算已采
            todo = [t for t in tids if t not in done]
            if args.limit:
                todo = todo[:args.limit]
            print(f"共 {len(tids)} 个 tid，待抓 {len(todo)} 个")

            ok = empty = fail = 0
            with out_path.open("a", encoding="utf-8") as f:
                for i, tid in enumerate(todo, 1):
                    rec = fetch_thread(driver, tid, args.delay)
                    if rec is None:
                        fail += 1
                        print(f"[{kw} {i}/{len(todo)}] {tid} 抓取失败", flush=True)
                    else:
                        rec["id"] = f"tb_{tid}"
                        rec["source"] = "tieba"
                        rec["keyword"] = kw
                        rec["crawled_at"] = datetime.now().strftime("%Y-%m-%d")
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        if rec.get("text"):
                            ok += 1
                            print(f"[{kw} {i}/{len(todo)}] {tid} {len(rec['text'])}字 "
                                  f"| {rec['title'][:30]}", flush=True)
                        else:
                            empty += 1
                            print(f"[{kw} {i}/{len(todo)}] {tid} 无正文"
                                  f"（{rec.get('note')}）", flush=True)
                    sleep_a_bit(args.delay)
            print(f"【{kw}】有正文 {ok}，无正文 {empty}，失败 {fail}")
            total_ok += ok
            total_empty += empty
            total_fail += fail
    finally:
        driver.quit()

    print(f"\n全部完成：有正文 {total_ok}，无正文 {total_empty}，失败 {total_fail}")
    print(f"输出：{out_path}")


if __name__ == "__main__":
    main()
