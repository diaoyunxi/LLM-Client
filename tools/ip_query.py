#!/usr/bin/env python3
# TOOL_NAME: ip_query
# TOOL_DESCRIPTION: 查询指定 IP 地址或域名的地理位置信息（国家、城市、ISP）
# TOOL_PARAMETERS: {"target": {"type": "string", "description": "IP 地址或域名", "required": true}}
"""
IP 地理位置查询工具
通过 ip-api.com 免费 API 查询 IP/域名的地理位置信息
"""

import json
import urllib.request
import urllib.error


def run(target: str) -> str:
    """查询 IP 或域名的地理位置"""
    url = f"http://ip-api.com/json/{target}?lang=zh-CN&fields=status,country,regionName,city,isp,org,query"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-Client/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") == "success":
            return json.dumps({
                "ip": data.get("query", target),
                "country": data.get("country", ""),
                "region": data.get("regionName", ""),
                "city": data.get("city", ""),
                "isp": data.get("isp", ""),
                "org": data.get("org", ""),
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"error": f"查询失败: {data.get('status', 'unknown')}"}, ensure_ascii=False)
    except urllib.error.URLError as e:
        return json.dumps({"error": f"网络请求失败: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
