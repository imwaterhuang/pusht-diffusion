"""Command line interface: audit, scenes, evaluation, reports, replay and local UI."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import default_config, load_config
from .utils import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pusht-diffusion')
    sub = parser.add_subparsers(dest='command', required=True)

    # 数据审计和场景生成。
    audit = sub.add_parser('audit')
    audit.add_argument('--dataset', required=True)
    audit.add_argument('--output', required=True)
    scenes = sub.add_parser('scenes')
    scenes.add_argument('--split', choices=['development', 'debug', 'final'], required=True)
    scenes.add_argument('--output', required=True)
    scenes.add_argument('--exclude', action='append', default=[])
    scenes.add_argument('--seed', type=int, default=20260924)
    scenes.add_argument('--count', type=int)
    scenes.add_argument('--selection')
    scenes.add_argument('--audit')

    # 固定模型选择后执行评测、生成报告或回放视频。
    lock = sub.add_parser('lock-selection')
    for flag in ('unet', 'dit', 'act', 'audit', 'development', 'output'):
        lock.add_argument('--' + flag, required=True)
    ev = sub.add_parser('eval')
    ev.add_argument('--model', choices=['act', 'unet', 'dit'], required=True)
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--audit', required=True)
    ev.add_argument('--scenes', required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--smoke-steps', type=int)
    report = sub.add_parser('report')
    report.add_argument('results', nargs='+')
    report.add_argument('--output', required=True)
    video = sub.add_parser('video')
    video.add_argument('--results', required=True)
    video.add_argument('--scene', required=True)
    video.add_argument('--output', required=True)

    # 交互界面、训练入口和代码归属检查。
    serve = sub.add_parser('serve')
    serve.add_argument('--act', required=True)
    serve.add_argument('--audit', required=True)
    serve.add_argument('--unet')
    serve.add_argument('--dit')
    serve.add_argument('--device', default='cpu')
    serve.add_argument('--port', type=int, default=7861)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--paths', required=True)
    train.add_argument('--resume')
    train.add_argument('--device', choices=('cpu', 'mps', 'cuda'))
    train.add_argument('--max-updates', type=int)
    own = sub.add_parser('check-ownership')
    args = parser.parse_args(argv)

    # 按子命令调用对应模块，训练仍使用 learner 中的实现。
    if args.command == 'audit':
        from .data import audit_data

        result = audit_data(args.dataset)
        atomic_json(args.output, result, frozen=True)
    elif args.command == 'scenes':
        from .scenes import collect_exclusions, generate_scenes, validate_selection

        selection = None
        excluded = list(args.exclude)
        if args.split == 'final':
            if not args.selection or not args.audit:
                parser.error('Final scenes require --selection and --audit after model selection')
            selection = validate_selection(args.selection, read(args.audit))
            excluded.append(selection['development_path'])
            # Include this project's debug records automatically, including interactive scene exclusions.
            excluded.extend(str(p) for p in Path('scenes').glob('*debug*.json*'))
        count = args.count or (50 if args.split == 'development' else 100 if args.split == 'final' else 1)
        result = generate_scenes(
            args.output,
            split=args.split,
            count=count,
            generation_seed=args.seed,
            exclusions=collect_exclusions(excluded),
            selection=selection,
        )
    elif args.command == 'lock-selection':
        from .scenes import lock_selection

        result = lock_selection(
            args.output,
            unet=args.unet,
            dit=args.dit,
            act=args.act,
            audit=read(args.audit),
            development=args.development,
        )
    elif args.command in ('eval', 'serve'):
        import torch

        torch.set_num_threads(2)
        from .policies import load_act_policy, load_diffusion_policy

        audit = read(args.audit)

        def load(name, path):
            return (
                load_act_policy(path, audit=audit, device=args.device)
                if name == 'act'
                else load_diffusion_policy(path, config=default_config(name), audit=audit, device=args.device)
            )

        if args.command == 'serve':
            from .live import PolicyRegistry, create_app
            import uvicorn

            policies = {'act': load('act', args.act)}
            for name in ('unet', 'dit'):
                if getattr(args, name):
                    policies[name] = load(name, getattr(args, name))
            uvicorn.run(
                create_app(PolicyRegistry(policies, debug_ledger='scenes/interactive-debug.jsonl')),
                host='127.0.0.1',
                port=args.port,
            )
            return
        from .evaluation import evaluate
        from .scenes import load_scenes

        result = evaluate(
            load(args.model, args.checkpoint),
            load_scenes(args.scenes),
            args.output,
            max_steps=args.smoke_steps or 300,
            evidence_kind='real_smoke' if args.smoke_steps else 'formal',
        )
    elif args.command == 'report':
        from .evaluation import render_report

        result = render_report(args.results, args.output)
    elif args.command == 'video':
        from .evaluation import replay_video

        result = replay_video(args.results, args.scene, args.output)
    elif args.command == 'train':
        from .learner.training import train

        train(load_config(args.config), read(args.paths), resume=args.resume,
              device=args.device, max_updates=args.max_updates)
        return
    else:
        from .ownership import check_ownership

        result = check_ownership()
    print(json.dumps(result.get('summary', result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
