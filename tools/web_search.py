"""
TOOL_NAME: web_search
TOOL_DESCRIPTION: 联网搜索，通过 DuckDuckGo 获取搜索结果，返回标题、摘要和链接
TOOL_PARAMETERS:
    query:
        type: string
        description: 搜索关键词，如 "Python 最新版本" 或 "how to use asyncio"
        required: true
    max_results:
        type: integer
        description: 返回的最大结果数量
        required: false
        default: 5
"""

import requests
import re
import json
import logging
from urllib.parse import quote_plus, unquote


logger = logging.getLogger("web_search")


def _search_duckduckgo(query: str, max_results: int = 5) -> list:
    """
    通过 DuckDuckGo HTML 版本获取搜索结果
    无需 API Key，无依赖
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    results = []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=headers, timeout=15, verify=True)
        resp.raise_for_status()
        html = resp.text

        # 提取搜索结果：DuckDuckGo HTML 版的结果在 class="result" 中
        # 使用正则匹配结果块
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL
        )

        if not result_blocks:
            # 备用匹配模式
            result_blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                html,
                re.DOTALL
            )
            for title, snippet in result_blocks[:max_results]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                if clean_title:
                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "url": "",
                    })
        else:
            for link, title, snippet in result_blocks[:max_results]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                # DuckDuckGo 的链接是跳转链接，提取实际 URL
                actual_url = ""
                match = re.search(r'uddg=([^&]+)', link)
                if match:
                    actual_url = unquote(match.group(1))
                results.append({
                    "title": clean_title,
                    "snippet": clean_snippet,
                    "url": actual_url,
                })

    except requests.RequestException as e:
        logger.warning("DuckDuckGo HTML 搜索失败: %s", e)
        return []

    return results


def _search_ddg_api(query: str, max_results: int = 5) -> list:
    """
    通过 DuckDuckGo Instant Answer API 获取摘要
    作为备用搜索方式
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        results = []

        # Abstract（即时回答）
        abstract = data.get("Abstract", "")
        if abstract:
            results.append({
                "title": data.get("Heading", ""),
                "snippet": abstract,
                "url": data.get("AbstractURL", ""),
                "source": "DuckDuckGo Instant Answer",
            })

        # RelatedTopics
        topics = data.get("RelatedTopics", [])
        for topic in topics[:max_results]:
            if isinstance(topic, dict):
                text = topic.get("Text", "")
                if text:
                    results.append({
                        "title": text[:80],
                        "snippet": text,
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo Related",
                    })

        return results
    except Exception as e:
        logger.warning("DuckDuckGo API 搜索失败: %s", e)
        return []


def run(query: str, max_results: int = 5):
    """
    执行联网搜索
    优先使用 DuckDuckGo HTML 搜索，失败时降级到 API
    """
    # 主搜索
    results = _search_duckduckgo(query, max_results)

    # 如果主搜索无结果，尝试 API
    if not results:
        results = _search_ddg_api(query, max_results)

    if not results:
        return {
            "query": query,
            "results": [],
            "total": 0,
            "message": "未找到搜索结果，可能是网络问题或搜索限制",
        }

    # 限制返回数量
    results = results[:max_results]

    return {
        "query": query,
        "total": len(results),
        "results": results,
    }