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
import logging


logger = logging.getLogger("shell_exec")

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

# 允许执行的命令白名单 (基于命令首个 token 匹配)
# 使用 shell=False 后, 不再支持管道/重定向, 仅允许独立命令
#
# 安全策略：不包含解释器/编译器（python, node, go, java 等），
# 因为它们可通过 -c / -e 参数执行任意代码，完全绕过白名单限制（CWE-78）。
# 例如: python3 -c "import os; os.system('rm -rf /')" 可绕过白名单执行任意命令。
ALLOWED_COMMANDS = {
    # 文件与目录查看
    "ls", "cat", "echo", "pwd", "head", "tail", "wc", "sort", "uniq",
    "cut", "tr", "diff", "find", "grep", "egrep", "fgrep", "which",
    "whereis", "file", "stat", "du", "basename", "dirname", "realpath",
    "readlink",
    # 系统信息
    "whoami", "hostname", "date", "uname", "id", "groups", "uptime",
    "w", "last", "env", "printenv", "df", "free", "ps", "lscpu",
    "lsmem", "lsblk",
    # 网络诊断（只读）
    "ping", "nslookup", "dig", "ifconfig", "ip", "netstat", "ss",
    "lsof", "dmesg",
    # 版本控制（只读操作安全）
    "git",
    # 文件操作
    "mkdir", "touch", "cp", "mv", "ln", "tar", "zip", "unzip",
    "gzip", "gunzip", "sed", "awk", "xargs", "tee", "seq", "yes",
    "test", "expr", "bc", "cal",
}

# 已知可通过 -c / -e 等参数执行任意代码的解释器/编译器，
# 即使通过其他途径出现在命令中也需要额外检查
_INTERPRETER_COMMANDS = frozenset({
    "python", "python3", "python3.11", "python3.12", "python3.10",
    "pip", "pip3", "node", "npm", "npx", "go", "java", "javac",
    "ruby", "perl", "php", "lua", "bash", "sh", "zsh",
    "mvn", "gradle", "cargo", "rustc", "make", "cmake",
    "curl", "wget",  # 可下载并管道执行任意脚本
})

# 命令最大长度限制 (字符)
MAX_COMMAND_LENGTH = 1000


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


def _check_whitelist(command: str) -> tuple[bool, str]:
    """
    检查命令是否在白名单中 (基于首个 token)
    返回 (是否允许, 原因)

    安全防护层级:
    1. 基础名匹配白名单（去除路径前缀）
    2. 解释器/编译器命令额外拦截（防止 -c/-e 参数绕过白名单执行任意代码）
    """
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return False, "命令解析失败"
    if not tokens:
        return False, "命令为空"
    # 取首个 token 的基础名 (去掉路径前缀, 如 /usr/bin/ls -> ls)
    base_cmd = os.path.basename(tokens[0])
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"命令 '{base_cmd}' 不在允许的白名单中"
    # 二次检查：即使基础名在白名单中，若属于解释器类命令仍需拒绝
    # 防止攻击者通过 python3 -c / node -e 等方式执行任意代码 (CWE-78)
    if base_cmd in _INTERPRETER_COMMANDS:
        return False, (
            f"命令 '{base_cmd}' 属于解释器/编译器，"
            "可通过 -c/-e 参数执行任意代码，出于安全考虑已禁止"
        )
    return True, ""


def run(command: str, timeout: int = 30, working_dir: str = ".", max_output: int = 10000):
    """
    执行 Shell 命令并返回结果

    安全措施:
    1. 命令长度限制 (1000 字符)
    2. 危险命令黑名单
    3. 命令白名单机制 (仅允许安全命令)
    4. 使用 shell=False + shlex.split() 防止命令注入
    """
    # 命令长度校验
    if len(command) > MAX_COMMAND_LENGTH:
        return {
            "success": False,
            "error": f"命令长度超过限制 ({MAX_COMMAND_LENGTH} 字符)",
            "output": "",
            "exit_code": -1,
        }

    # 危险命令黑名单检查
    is_dangerous, reason = _is_dangerous(command)
    if is_dangerous:
        return {
            "success": False,
            "error": f"安全限制: {reason}",
            "output": "",
            "exit_code": -1,
        }

    # 白名单检查
    allowed, reason = _check_whitelist(command)
    if not allowed:
        return {
            "success": False,
            "error": f"安全限制: {reason}",
            "output": "",
            "exit_code": -1,
        }

    # 使用 shlex.split() 解析命令, shell=False 防止命令注入
    try:
        args_list = shlex.split(command)
        if not args_list:
            return {
                "success": False,
                "error": "命令为空",
                "output": "",
                "exit_code": -1,
            }
    except ValueError as e:
        return {
            "success": False,
            "error": "命令解析错误",
            "output": "",
            "exit_code": -1,
        }

    cwd = working_dir if os.path.isdir(working_dir) else "."
    env = {**os.environ, "TERM": "dumb"}

    try:
        process = subprocess.run(
            args_list,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
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
            stderr = stderr[:max_output // 2] + "\n... (错误输出已截断)"
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
    except FileNotFoundError:
        return {
            "success": False,
            "error": "命令未找到",
            "output": "",
            "exit_code": -1,
        }
    except Exception as e:
        logger.warning("shell_exec 执行失败: %s", e)
        return {
            "success": False,
            "error": "执行失败",
            "output": "",
            "exit_code": -1,
        }
