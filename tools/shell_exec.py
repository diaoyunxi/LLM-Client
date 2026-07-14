"""
TOOL_NAME: shell_exec
TOOL_DESCRIPTION: 在本地执行 Shell 命令并返回输出结果。支持超时控制，限制最大输出长度。请谨慎使用，仅执行安全命令。
TOOL_PARAMETERS:
    command:
        type: string
        description: 要执行的 Shell 命令，如 "ls -la" 或 "python --version"
        required: true
    timeout:
        type: integer
        description: 命令执行超时时间（秒），超时后自动终止
        required: false
        default: 30
    working_dir:
        type: string
        description: 命令执行的工作目录，默认为当前目录
        required: false
        default: .
    max_output:
        type: integer
        description: 输出结果最大字符数，超出部分会被截断
        required: false
        default: 10000
"""

import subprocess
import os
import shlex


# 危险命令黑名单
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "fork bomb",
    "> /dev/sda",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
]


def _is_dangerous(command: str) -> tuple[bool, str]:
    """
    检查命令是否在危险黑名单中
    返回 (是否危险, 原因)
    """
    cmd_lower = command.lower().strip()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous.lower() in cmd_lower:
            return True, f"命令包含危险操作: {dangerous}"
    return False, ""


def run(command: str, timeout: int = 30, working_dir: str = ".", max_output: int = 10000):
    """
    执行 Shell 命令并返回结果
    """
    # 安全检查
    is_dangerous, reason = _is_dangerous(command)
    if is_dangerous:
        return {
            "success": False,
            "error": f"安全限制: {reason}",
            "output": "",
            "exit_code": -1,
        }

    # 解析命令
    try:
        # 使用 shell=True 以支持管道、重定向等
        args = {
            "shell": True,
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "cwd": working_dir if os.path.isdir(working_dir) else ".",
            "env": {**os.environ, "TERM": "dumb"},
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"参数错误: {e}",
            "output": "",
            "exit_code": -1,
        }

    try:
        process = subprocess.run(
            command,
            **args,
        )
        stdout = process.stdout or ""
        stderr = process.stderr or ""

        # 截断过长输出
        stdout_truncated = False
        stderr_truncated = False
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + f"\n... (输出已截断，共 {len(process.stdout or '')} 字符)"
            stdout_truncated = True
        if len(stderr) > max_output // 2:
            stderr = stderr[:max_output // 2] + f"\n... (错误输出已截断)"
            stderr_truncated = True

        return {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"命令执行超时 ({timeout} 秒)",
            "output": "",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "output": "",
            "exit_code": -1,
        }