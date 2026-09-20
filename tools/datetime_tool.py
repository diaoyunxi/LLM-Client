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

from datetime import datetime
import time


def run(format: str = "iso", timezone: str = "Asia/Shanghai"):
    """
    获取当前日期时间
    """
    now = datetime.now()

    # 处理时区（简化处理，实际应用可用 pytz 或 zoneinfo）
    tz_offsets = {
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

    offset_hours = tz_offsets.get(timezone, 8)
    # 这里简化处理，实际应使用 zoneinfo
    from datetime import timezone as dt_timezone, timedelta
    tz = dt_timezone(timedelta(hours=offset_hours))
    now_with_tz = now.replace(tzinfo=tz)

    # 格式化输出
    if format.lower() == "iso":
        formatted = now_with_tz.isoformat()
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
