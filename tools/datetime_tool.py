"""
TOOL_NAME: datetime_tool
TOOL_DESCRIPTION: 获取当前日期和时间信息，支持格式化输出和时区转换
TOOL_PARAMETERS:
    format:
        type: string
        description: 日期时间格式，如 "%Y-%m-%d %H:%M:%S" 或 "iso"、"timestamp"、"date"、"time"
        required: false
        default: iso
    timezone:
        type: string
        description: 时区名称，如 "Asia/Shanghai"、"UTC"、"America/New_York"
        required: false
        default: Asia/Shanghai
"""

from datetime import datetime, timezone as dt_timezone, timedelta
import time


# 时区偏移映射（小时）
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
    """
    offset_hours = _TZ_OFFSETS.get(timezone, 8)
    tz = dt_timezone(timedelta(hours=offset_hours))

    # 使用 timezone-aware 的 now()，避免依赖服务器本地时区
    # 先获取 UTC 时间，再转换为目标时区，确保跨平台一致性
    now = datetime.now(tz=dt_timezone.utc).astimezone(tz)

    # 格式化输出
    if format.lower() == "iso":
        formatted = now.isoformat()
    elif format.lower() == "timestamp":
        formatted = str(int(now.timestamp()))
    elif format.lower() == "date":
        formatted = now.strftime("%Y-%m-%d")
    elif format.lower() == "time":
        formatted = now.strftime("%H:%M:%S")
    else:
        formatted = now.strftime(format)

    return {
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
