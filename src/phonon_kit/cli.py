from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .errors import PhononKitError
from .initializer import init_case
from .runner import collect_config, replot_config, resume_config, run_config, status_text
from .state import RunPaths, latest_run
from .util import load_json
from .validation import validate_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ph", description="DeepMD/VASP finite-displacement phonon workflow")
    parser.add_argument("--version", action="version", version=f"phonon-kit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="创建一个清晰的声子计算案例")
    init.add_argument("target", type=Path)

    validate = sub.add_parser("validate", help="检查配置、依赖、模型和 VASP/调度输入")
    validate.add_argument("config", type=Path)
    validate.add_argument("--only", nargs="+", default=None)

    run = sub.add_parser("run", help="开始或自动恢复运行")
    run.add_argument("config", type=Path)
    run.add_argument("--only", nargs="+", default=None)
    run.add_argument("--new", action="store_true", help="保留已有现场并强制创建新版本")
    run.add_argument("--wait", action="store_true", help="等待 DFT 完成")

    resume = sub.add_parser("resume", help="恢复与当前配置匹配的运行")
    resume.add_argument("config", type=Path)
    resume.add_argument("--wait", action="store_true", help="持续等待 DFT 完成")

    collect = sub.add_parser("collect", help="收集已经放回本地的 VASP 结果")
    collect.add_argument("config", type=Path)

    status = sub.add_parser("status", help="显示最新运行状态")
    status.add_argument("target", type=Path, help="config.yaml 或具体运行目录")

    plot = sub.add_parser("plot", help="从已有结果重新生成 PNG")
    plot.add_argument("config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            target = init_case(args.target)
            print(f"案例已创建: {target}")
            print(f"下一步: cd {target} && ph validate config.yaml")
            return 0
        if args.command == "status" and args.target.is_dir():
            paths = RunPaths(args.target.resolve())
            state = load_json(paths.state_file)
            print(status_text(paths, state))
            return 0

        config_path = args.target if args.command == "status" else args.config
        config = load_config(config_path)
        if args.command == "validate":
            report = validate_runtime(config, args.only, real_inference=True)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print("验证通过")
            return 0
        if args.command == "run":
            paths, state, created = run_config(config, only=args.only, force_new=args.new, wait=args.wait)
            print(("新建运行: " if created else "使用运行: ") + str(paths.root))
            print(status_text(paths, state))
            return 0
        if args.command == "resume":
            paths, state = resume_config(config, wait=args.wait)
            print(status_text(paths, state))
            return 0
        if args.command == "collect":
            paths, state = collect_config(config)
            print(status_text(paths, state))
            return 0
        if args.command == "plot":
            paths, state = replot_config(config)
            print(f"PNG 已重新生成: {paths.results}")
            print(status_text(paths, state))
            return 0
        if args.command == "status":
            latest = latest_run(config)
            if latest is None:
                print("尚无运行")
                return 0
            print(status_text(*latest))
            return 0
        raise AssertionError(args.command)
    except (PhononKitError, ValueError, OSError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
