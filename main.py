#!/usr/bin/env python3
"""
LLM 客户端统一入口
支持三种界面模式：CLI（命令行）、TUI（终端界面）、GUI（图形界面）

用法:
    python main.py --mode cli          # 纯终端命令行
    python main.py --mode tui          # Textual TUI 界面
    python main.py --mode gui          # PyQt6 GUI 界面

    python main.py --backend ollama --host localhost --port 11434
    python main.py --backend llamacpp --host localhost --port 8080
    python main.py --backend openai --base-url https://api.openai.com/v1 --api-key sk-xxx
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="LLM 客户端 - 支持 Ollama / llama.cpp / OpenAI API / 多模态 / 工具调用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --mode cli --backend ollama --model llama3.1
  %(prog)s --mode gui --backend llamacpp --host 192.168.1.100 --port 8080
  %(prog)s --mode cli --image ./photo.png "描述这张图片"
  %(prog)s --backend openai --base-url https://api.openai.com/v1 --api-key sk-xxx --model gpt-4o
        """
    )

    parser.add_argument(
        "--mode",
        choices=["cli", "tui", "gui"],
        default="cli",
        help="界面模式：cli=纯终端，tui=终端界面 (textual), gui=图形界面 (PyQt6) (默认：cli)",
    )

    parser.add_argument(
        "--backend",
        choices=["ollama", "llamacpp", "openai"],
        default="ollama",
        help="后端类型：ollama / llamacpp / openai (默认：ollama)",
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="后端主机地址 (默认：localhost)，OpenAI 后端使用 --base-url",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="后端端口 (ollama 默认：11434, llamacpp 默认：8080)",
    )

    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI 兼容 API 的完整 URL（如 https://api.openai.com/v1），仅 openai 后端使用",
    )

    parser.add_argument(
        "--api-key",
        default="",
        help="API Key（仅 openai 后端需要）",
    )

    parser.add_argument(
        "--model",
        default="",
        help="模型名称 (如 llama3.1, qwen2.5, gpt-4o 等)",
    )

    parser.add_argument(
        "--tools-dir",
        default="tools",
        help="外置工具目录 (默认：tools)",
    )

    parser.add_argument(
        "--system",
        default="",
        help="系统提示词",
    )

    parser.add_argument(
        "--image",
        default="",
        help="上传图片路径（多模态模型用）",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度 (默认：0.7)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="智能体最大迭代次数 (默认：10)",
    )

    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="启用模型思考过程显示（需模型支持，如 DeepSeek-R1、Qwen3）",
    )

    args = parser.parse_args()

    # 根据模式启动对应界面
    if args.mode == "cli":
        from interfaces.cli import run_cli
        # 直接传递参数，不重写 sys.argv
        run_cli(
            backend=args.backend,
            host=args.host,
            port=args.port,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            tools_dir=args.tools_dir,
            system=args.system,
            image=args.image,
        )

    elif args.mode == "tui":
        from interfaces.tui import run_tui
        run_tui()

    elif args.mode == "gui":
        from interfaces.gui import run_gui
        run_gui()

    else:
        print(f"未知模式：{args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
