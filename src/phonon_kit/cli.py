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

    qha = sub.add_parser("qha", help="多相准谐近似和 P-T 相图工作流")
    qha_sub = qha.add_subparsers(dest="qha_command", required=True)
    qha_init = qha_sub.add_parser("init", help="创建多相 QHA 案例")
    qha_init.add_argument("target", type=Path)
    qha_init.add_argument("--phases", nargs="+", required=True, help="至少两个晶相名称")
    qha_validate = qha_sub.add_parser("validate", help="检查 QHA 配置、模型、结构和 VASP 模板")
    qha_validate.add_argument("config", type=Path)
    qha_validate.add_argument("--only", nargs="+", default=None)
    qha_plan = qha_sub.add_parser("plan", help="预览体积、位移和任务数量，不开始计算")
    qha_plan.add_argument("config", type=Path)
    qha_run = qha_sub.add_parser("run", help="开始或自动恢复多相 QHA")
    qha_run.add_argument("config", type=Path)
    qha_run.add_argument("--only", nargs="+", default=None)
    qha_run.add_argument("--new", action="store_true", help="保留现场并创建新版本")
    qha_run.add_argument("--wait", action="store_true", help="持续等待 DFT 各阶段完成")
    qha_resume = qha_sub.add_parser("resume", help="恢复匹配的 QHA 运行")
    qha_resume.add_argument("config", type=Path)
    qha_resume.add_argument("--wait", action="store_true", help="持续等待 DFT 各阶段完成")
    qha_collect = qha_sub.add_parser("collect", help="收集手动放回的 QHA VASP 结果")
    qha_collect.add_argument("config", type=Path)
    qha_status = qha_sub.add_parser("status", help="显示 QHA 运行状态")
    qha_status.add_argument("target", type=Path, help="qha.yaml 或具体 QHA 运行目录")
    qha_plot = qha_sub.add_parser("plot", help="从现有 QHA 数值结果重新绘图")
    qha_plot.add_argument("config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "qha":
            from .qha_config import load_qha_config
            from .qha_initializer import init_qha_case
            from .qha_runner import (
                collect_qha_config,
                qha_status_text,
                replot_qha_config,
                resume_qha_config,
                run_qha_config,
            )
            from .qha_state import QHARunPaths, latest_qha_run
            from .qha_validation import qha_plan, validate_qha_runtime

            if args.qha_command == "init":
                target = init_qha_case(args.target, args.phases)
                print(f"QHA 案例已创建: {target}")
                print(f"下一步: cd {target} && ph qha plan qha.yaml")
                return 0
            if args.qha_command == "status" and args.target.is_dir():
                paths = QHARunPaths(args.target.resolve())
                print(qha_status_text(paths, load_json(paths.state_file)))
                return 0
            config_path = args.target if args.qha_command == "status" else args.config
            config = load_qha_config(config_path)
            if args.qha_command == "plan":
                print(json.dumps(qha_plan(config), indent=2, ensure_ascii=False))
                return 0
            if args.qha_command == "validate":
                print(json.dumps(validate_qha_runtime(config, args.only, real_inference=True), indent=2, ensure_ascii=False))
                print("QHA 验证通过")
                return 0
            if args.qha_command == "run":
                paths, state, created = run_qha_config(config, only=args.only, force_new=args.new, wait=args.wait)
                print(("新建 QHA 运行: " if created else "使用 QHA 运行: ") + str(paths.root))
                print(qha_status_text(paths, state))
                return 0
            if args.qha_command == "resume":
                paths, state = resume_qha_config(config, wait=args.wait)
                print(qha_status_text(paths, state))
                return 0
            if args.qha_command == "collect":
                paths, state = collect_qha_config(config)
                print(qha_status_text(paths, state))
                return 0
            if args.qha_command == "plot":
                paths, state = replot_qha_config(config)
                print(f"QHA PNG 已重新生成: {paths.results}")
                print(qha_status_text(paths, state))
                return 0
            if args.qha_command == "status":
                latest = latest_qha_run(config)
                if latest is None:
                    print("尚无 QHA 运行")
                    return 0
                print(qha_status_text(*latest))
                return 0
            raise AssertionError(args.qha_command)
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
