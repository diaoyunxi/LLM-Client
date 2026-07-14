"""
纯终端命令行界面
最简洁的交互方式，无需额外依赖
"""

import os
import sys
import json
import argparse
from typing import Optional

# 将上级目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backend import OllamaBackend, LlamaCppBackend
from core.conversation import Conversation
from core.agent import AgentLoop
from core.tools.loader import ToolLoader


def print_banner():
    print("=" * 60)
    print("   LLM 客户端 - 命令行界面")
    print("   支持 Ollama / llama.cpp | 多模态 | 工具调用")
    print("=" * 60)
    print()


def select_model(backend) -> Optional[str]:
    """交互式选择模型"""
    models = backend.list_models()
    if not models:
        print("未检测到模型，请确认后端服务已启动。")
        return None

    print("可用模型:")
    for i, m in enumerate(models, 1):
        vision = " [视觉]" if m.supports_vision else ""
        tools = " [工具]" if m.supports_tools else ""
        print(f"  {i}. {m.name}{vision}{tools}")
    print()

    while True:
        choice = input("请选择模型编号 (或输入模型名): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx].name
        else:
            # 直接输入模型名
            for m in models:
                if m.name == choice or m.name.startswith(choice):
                    return m.name
        print("无效选择，请重试。")


def run_cli(backend: str = None, host: str = None, port: int = None,
            model: str = None, tools_dir: str = None, system: str = None,
            image: str = None):
    """
    启动 CLI 界面

    支持两种调用方式:
    1. 直接传递关键字参数 (由 main.py 调用, 不重写 sys.argv)
    2. 不传参数时从命令行解析 (兼容独立运行 python -m interfaces.cli)

    Args:
        backend: 后端类型 (ollama / llamacpp)
        host: 后端主机地址
        port: 后端端口
        model: 模型名称
        tools_dir: 工具目录
        system: 系统提示词
        image: 图片路径
    """
    # 若未通过参数传入, 则从命令行解析 (兼容直接运行)
    if backend is None:
        parser = argparse.ArgumentParser(description="LLM 客户端 CLI")
        parser.add_argument("--backend", choices=["ollama", "llamacpp"], default="ollama",
                            help="后端类型")
        parser.add_argument("--host", default="localhost", help="后端主机地址")
        parser.add_argument("--port", type=int, default=None, help="后端端口")
        parser.add_argument("--model", default="", help="模型名称")
        parser.add_argument("--tools-dir", default="tools", help="工具目录")
        parser.add_argument("--system", default="", help="系统提示词")
        parser.add_argument("--image", default="", help="上传图片路径（多模态）")
        args = parser.parse_args()
    else:
        # 使用直接传入的参数
        from types import SimpleNamespace
        args = SimpleNamespace(
            backend=backend,
            host=host or "localhost",
            port=port,
            model=model or "",
            tools_dir=tools_dir or "tools",
            system=system or "",
            image=image or "",
        )

    print_banner()

    # 确定端口
    port = args.port or (11434 if args.backend == "ollama" else 8080)

    # 初始化后端
    if args.backend == "ollama":
        backend = OllamaBackend(host=args.host, port=port)
        print(f"[后端] Ollama @ {args.host}:{port}")
    else:
        backend = LlamaCppBackend(host=args.host, port=port)
        print(f"[后端] llama.cpp @ {args.host}:{port}")

    # 加载工具
    tool_loader = ToolLoader()
    tools_dir = os.path.abspath(args.tools_dir)
    if os.path.isdir(tools_dir):
        tool_loader.add_tools_dir(tools_dir)
        loaded = tool_loader.load_all()
        print(f"[工具] 已加载 {loaded} 个工具")
    else:
        print(f"[工具] 目录不存在: {tools_dir}")

    # 选择模型
    model = args.model
    if not model:
        model = select_model(backend)
        if not model:
            print("未选择模型，退出。")
            return
    print(f"[模型] {model}\n")

    # 初始化对话
    conversation = Conversation(id="cli_session", title="CLI 对话", model=model)
    if args.system:
        conversation.add_system_message(args.system)
        print(f"[系统提示] {args.system}\n")

    # 智能体循环
    agent = AgentLoop(
        backend=backend,
        conversation=conversation,
        tool_loader=tool_loader,
        model=model,
        max_iterations=10,
    )

    print("输入 '/help' 查看命令，'/quit' 退出。\n")

    # 预加载图片
    images = []
    if args.image:
        if os.path.exists(args.image):
            images.append(backend.encode_image(args.image))
            print(f"[图片] 已加载: {args.image}\n")
        else:
            print(f"[警告] 图片不存在: {args.image}")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            # 添加空列表检查, 防止用户仅输入 "/" 时 split() 返回空列表导致 IndexError
            parts = user_input[1:].lower().split()
            if not parts:
                print("[系统] 请输入有效命令，输入 /help 查看帮助。")
                continue
            cmd = parts[0]
            if cmd in ("quit", "exit", "q"):
                print("再见！")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "clear":
                conversation.clear()
                print("[系统] 对话历史已清空。")
            elif cmd == "save":
                parts = user_input.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else f"conversation_{conversation.id}.json"
                conversation.save(path)
                print(f"[系统] 对话已保存到: {path}")
            elif cmd == "load":
                parts = user_input.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else None
                if path and os.path.exists(path):
                    conversation = Conversation.load(path)
                    agent.conversation = conversation
                    print(f"[系统] 对话已加载: {path}")
                else:
                    print("[系统] 请指定有效的文件路径。")
            elif cmd == "image":
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and os.path.exists(parts[1]):
                    images.append(backend.encode_image(parts[1]))
                    print(f"[图片] 已加载: {parts[1]}")
                else:
                    print("[系统] 请指定有效的图片路径。")
            elif cmd == "models":
                for m in backend.list_models():
                    print(f"  - {m.name}")
            elif cmd == "tools":
                for name in tool_loader.tools:
                    t = tool_loader.tools[name]
                    print(f"  - {name}: {t.description}")
            else:
                print(f"未知命令: {cmd}")
            continue

        # 发送消息
        print("AI > ", end="", flush=True)
        try:
            for chunk in agent.run(user_input, images=images, stream=True):
                print(chunk, end="", flush=True)
            print()  # 换行
        except Exception as e:
            print(f"\n[错误] {e}")

        # 清空图片（避免重复发送）
        images = []
        print()


def print_help():
    print("""
可用命令:
  /help          显示此帮助
  /quit, /exit   退出程序
  /clear         清空对话历史
  /save [路径]   保存对话到文件
  /load [路径]   从文件加载对话
  /image <路径>  加载图片（多模态）
  /models        列出可用模型
  /tools         列出已加载的工具
""")


if __name__ == "__main__":
    run_cli()
