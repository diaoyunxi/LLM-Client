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
    """查询 IP 地址的地理位置信息"""
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ip": ip,
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country_name"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "org": data.get("org"),
            "timezone": data.get("timezone"),
        }
    except requests.RequestException as e:
        return {"error": f"查询失败: {e}"}
    except Exception as e:
        return {"error": f"解析失败: {e}"}
