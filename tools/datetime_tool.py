"""
TOOL_NAME: datetime_tool
TOOL_DESCRIPTION: 获取当前日期和时间信息，支持格式化输出和时区转换
TOOL_PARAMETERS:
    format:
        type: string
        description: 日期时间格式，如 "%Y-%m-%d %H:%M:%S" 或 "iso"、"timestamp"
        required: false
        default: iso
    timezone:
        type: string
        description: 时区名称，如 "Asia/Shanghai"、"UTC"、"America/New_York"
        required: false
        default: Asia/Shanghai
"""

from datetime import datetime, timezone as dt_timezone, timedelta


# 常用时区偏移量（小时），未命中时回退到 UTC 而非默认 +8，避免跨时区静默错误
_TZ_OFFSETS = {
    "UTC": 0,
    "Asia/Shanghai": 8,
    "Asia/Tokyo": 9,
    "Asia/Seoul": 9,
    "Asia/Singapore": 8,
    "Asia/Hong_Kong": 8,
    "Asia/Bangkok": 7,
    "Asia/Dubai": 4,
    "Europe/London": 0,
    "Europe/Paris": 1,
    "Europe/Berlin": 1,
    "Europe/Moscow": 3,
    "America/New_York": -5,
    "America/Los_Angeles": -8,
    "America/Chicago": -6,
    "America/Denver": -7,
    "Australia/Sydney": 11,
}


def run(format: str = "iso", timezone: str = "Asia/Shanghai"):
    """
    获取当前日期时间

    修复 (CWE-682)：
    原实现先用 datetime.now()（naive），再 replace(tzinfo=...) 构造一个"带时区"
    的对象，但后续 weekday / strftime / timestamp 调用仍然使用原始的 naive now，
    当系统时区与请求时区不一致时，返回的 weekday、hour、timestamp 全部与
    声明的 timezone 字段矛盾（例如用户在纽约服务器上请求 Asia/Shanghai，
    会得到"上海时区"但"纽约星期几"的混乱结果）。

    修复后统一使用 timezone-aware 的 datetime.now(tz)，所有派生字段
    (weekday/hour/minute/second/timestamp/isoformat/strftime) 均来自同一对象，
    保证 timezone 字段与其他字段一致。未命中内置偏移表时回退 UTC（而非静默
    使用 +8），并在返回中标注 fallback 提示调用方校验。
    """
    offset_hours = _TZ_OFFSETS.get(timezone)
    fallback_used = offset_hours is None
    if fallback_used:
        offset_hours = 0  # 回退 UTC，避免静默采用 +8

    tz = dt_timezone(timedelta(hours=offset_hours))
    now = datetime.now(tz)  # timezone-aware，所有后续字段基于此对象

    # 格式化输出
    fmt_lower = format.lower()
    if fmt_lower == "iso":
        formatted = now.isoformat()
    elif fmt_lower == "timestamp":
        formatted = str(int(now.timestamp()))
    elif fmt_lower == "date":
        formatted = now.strftime("%Y-%m-%d")
    elif fmt_lower == "time":
        formatted = now.strftime("%H:%M:%S")
    else:
        formatted = now.strftime(format)

    result = {
        "datetime": formatted,
        "timezone": timezone,
        "utc_offset": f"UTC{offset_hours:+d}",
        "timestamp": int(now.timestamp()),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "weekday": now.strftime("%A"),
    }
    if fallback_used:
        result["warning"] = (
            f"timezone '{timezone}' 未在内置偏移表中命中，已回退 UTC；"
            "请传入标准时区名或使用 zoneinfo 以获得夏令时支持。"
        )
    return result
