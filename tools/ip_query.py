"""
TOOL_NAME: ip_query
TOOL_DESCRIPTION: 查询 IP 地址的地理位置信息
TOOL_PARAMETERS:
    ip:
        type: string
        description: 要查询的 IP 地址，如 "8.8.8.8"
        required: true
"""

import requests


def run(ip: str):
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=10)
        data = resp.json()
        return {
            "ip": ip,
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country_name"),
            "org": data.get("org"),
        }
    except Exception as e:
        return {"error": str(e), "ip": ip}
