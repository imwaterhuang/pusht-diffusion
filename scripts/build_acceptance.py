"""Validate a completed final-50 run and build a Chinese evidence report and videos.

No training, policy sampling, scene redraw or retrospective best-looking video
selection occurs here. Videos replay the already recorded physical actions.
"""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

from colab_evaluate import AMENDMENT, PROTOCOL, frozen_scenes
from pusht_diffusion.config import default_config
from pusht_diffusion.environment import SceneState, make_env
from pusht_diffusion.evaluation import checkpoint_score, episode_metrics, recompute, replay_video
from pusht_diffusion.utils import atomic_json, canonical_hash, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def validate_inputs(results: Path, scenes: Path, selection_path: Path) -> tuple[dict, dict, dict, list[dict]]:
    result, pool, selection = read(results), frozen_scenes(scenes), read(selection_path)
    if (pool.get("schema") != "pusht_scenes_v2" or pool.get("count") != 50
            or pool.get("split") != "final" or pool.get("protocol_amendment") != AMENDMENT
            or selection.get("schema") != "pusht_selection_v2" or pool["selection"] != selection
            or selection.get("lock_hash") != canonical_hash({k: v for k, v in selection.items() if k != "lock_hash"})
            or selection.get("exclusions_hash") != canonical_hash(pool["exclusions"])
            or selection.get("final_generation_seed") != pool.get("generation_seed")):
        raise ValueError("Final pool/selection lock mismatch")
    candidates = selection.get("candidates", [])
    if sorted(x["step"] for x in candidates) != list(range(5000, 40001, 5000)):
        raise ValueError("Expected all eight frozen development candidates")
    winner = max(candidates, key=lambda x: checkpoint_score(x["summary"], x["step"]))
    selected = selection["models"]["unet"]
    if (selection.get("chosen_sha256") != winner["checkpoint_sha256"]
            or selected.get("checkpoint_sha256") != winner["checkpoint_sha256"]
            or selection.get("chosen_step") != winner["step"]
            or selection.get("score_order") != ["success_count_87", "mean_max_coverage", "earlier_step"]):
        raise ValueError("Selected checkpoint is inconsistent with frozen development ranking")
    runner, provenance = result.get("runner", {}), result.get("provenance", {})
    if (result.get("schema") != "pusht_results_v1" or result.get("evidence_kind") != "formal"
            or result.get("policy") != "unet" or result.get("scene_split") != "final"
            or result.get("scenes_hash") != pool["scenes_hash"] or result.get("protocol") != PROTOCOL
            or runner.get("complete") is not True or runner.get("expected_scene_count") != 50
            or runner.get("protocol_amendment") != AMENDMENT or len(result.get("episodes", [])) != 50
            or provenance.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or provenance.get("step") != winner["step"] or provenance.get("weights") != "ema"
            or provenance.get("kind") != "trained_policy"
            or provenance.get("sampling") != default_config()["sampling"]
            or provenance.get("config_fingerprint") != canonical_hash(default_config())):
        raise ValueError("Incomplete or incompatible final evaluation")
    manifest_path = results.parent / "manifest.json"
    manifest = read(manifest_path)
    if (runner.get("manifest_sha256") != canonical_hash(manifest)
            or manifest.get("scenes_file_sha256") != file_hash(scenes)
            or manifest.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or manifest.get("runtime") != runner.get("runtime")
            or manifest.get("evaluator_code_fingerprint") != result.get("evaluator_code_fingerprint")):
        raise ValueError("Final evaluation manifest mismatch")
    rows = []
    for episode, scene in zip(result["episodes"], pool["scenes"], strict=True):
        if any(episode.get(key) != value for key, value in scene.items()):
            raise ValueError("Final scene order/state mismatch")
        metrics = episode_metrics(episode)
        if episode.get("metrics") != metrics or not 0 <= metrics["steps"] <= 300:
            raise ValueError("Episode cached metrics/length mismatch")
        if episode["trace"]:
            last = episode["trace"][-1]
            if not (metrics["steps"] == 300 or last["coverage"] > .95 or last["terminated"] or last["truncated"]):
                raise ValueError("Prematurely stopped episode")
        elif episode["initial_coverage"] <= .95:
            raise ValueError("Empty non-successful episode")
        rows.append({"scene_id": scene["scene_id"], "environment_seed": scene["environment_seed"],
                     "initial_coverage": episode["initial_coverage"], **metrics,
                     "terminal_reason": episode["terminal_reason"]})
    if recompute(result) != result.get("summary"):
        raise ValueError("Final summary differs from recomputed trace metrics")
    return result, pool, selection, rows


def choose_videos(rows: list[dict], selection: dict) -> dict:
    rule = selection.get("video_example_rule", {})
    if (rule.get("success_example") != "first scene_id in frozen order with success_95"
            or rule.get("failure_example") != "first scene_id in frozen order without success_87"):
        raise ValueError("Unknown predeclared video example rule")
    # Frozen IDs use zero-padded indices; require their order to equal lexical order.
    if [r["scene_id"] for r in rows] != sorted(r["scene_id"] for r in rows):
        raise ValueError("Frozen order and smallest-scene-id video rule disagree")
    return {"success": next((r["scene_id"] for r in rows if r["success_95"]), None),
            "failure": next((r["scene_id"] for r in rows if not r["success_87"]), None)}


def copy_evidence(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError(f"Existing evidence differs: {target.name}")
    else:
        shutil.copyfile(source, target)
    if file_hash(source) != file_hash(target):
        raise IOError("Evidence copy checksum mismatch")


def make_gif(video: Path, output: Path) -> dict:
    import av
    from PIL import Image

    frames = []
    next_time = 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / float(stream.average_rate)
            if timestamp + 1e-8 < next_time:
                continue
            picture = frame.to_image().convert("RGB")
            picture.thumbnail((320, 320), Image.Resampling.LANCZOS)
            frames.append(picture)
            next_time += .2
    if not frames:
        raise ValueError("Video has no decodable frames")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=False)
    return {"source_video_sha256": file_hash(video), "sha256": file_hash(output),
            "fps": 5, "max_size": [320, 320], "frames": len(frames),
            "method": "time-subsampled and resized deterministic replay; actions unchanged"}


def videos(result: dict, results_path: Path, choices: dict, output: Path) -> dict:
    expected = result["runner"]["runtime"]["packages"]
    for package in ("gym-pusht", "gymnasium", "pymunk", "shapely", "pygame", "numpy", "Pillow"):
        if version(package) != expected[package]:
            raise ValueError(f"Video replay runtime mismatch for {package}; use original evaluation environment")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for category, scene_id in choices.items():
        if scene_id is None:
            artifacts[category] = {"status": "category_absent", "scene_id": None}
            continue
        video = output / f"{category}.mp4"
        proof_path = video.with_suffix(".provenance.json")
        if video.exists():
            proof = read(proof_path)
            if (proof["scene_id"] != scene_id or proof["source_sha256"] != file_hash(results_path)
                    or proof["video_sha256"] != file_hash(video) or proof["max_coverage_error"] > 1e-7):
                raise ValueError("Existing video provenance differs from frozen selected trace")
        else:
            print(f"Replay {category}: {scene_id}", flush=True)
            proof = replay_video(results_path, scene_id, video)
        gif = video.with_suffix(".gif")
        gif_proof = make_gif(video, gif)
        atomic_json(gif.with_suffix(".gif.provenance.json"), gif_proof)
        artifacts[category] = {"status": "verified_replay", "scene_id": scene_id,
                               "video": str(video.resolve()), "gif": str(gif.resolve()),
                               "provenance": proof, "gif_provenance": gif_proof}
    return artifacts


def montage(result: dict, results_path: Path, output: Path) -> dict:
    """Five pages of ten exact trace replays, encoded incrementally at 10 fps."""
    import av
    from PIL import Image, ImageDraw

    destination = output / "all50.mp4"
    proof_path = destination.with_suffix(".provenance.json")
    identifiers = [episode["scene_id"] for episode in result["episodes"]]
    source_hash = file_hash(results_path)
    if destination.exists():
        proof = read(proof_path)
        if (proof.get("source_sha256") != source_hash or proof.get("scene_ids") != identifiers
                or proof.get("video_sha256") != file_hash(destination) or proof["max_coverage_error"] > 1e-7):
            raise ValueError("Existing montage provenance differs from all 50 frozen traces")
        return proof
    temporary = destination.with_name("all50.pending.mp4")
    width, height, tile_height = 960, 544, 256
    maximum_error, total_frames, pages = 0.0, 0, []
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": "22", "preset": "fast"}
            for page_index in range(5):
                print(f"Montage page {page_index + 1}/5: replaying ten frozen scenes", flush=True)
                group = result["episodes"][page_index * 10:(page_index + 1) * 10]
                environments, images = [], []
                page_error = 0.0
                try:
                    for episode in group:
                        env = make_env(300)
                        environments.append(env)
                        _, info = env.reset(seed=episode["environment_seed"], options={
                            "reset_to_state": SceneState(**episode["state"]).as_array()})
                        page_error = max(page_error, abs(float(info["coverage"]) - episode["initial_coverage"]))
                        images.append(Image.fromarray(np.asarray(env.render())).convert("RGB"))
                    longest = max(len(ep["trace"]) for ep in group)
                    for frame_step in range(longest + 1):
                        canvas = Image.new("RGB", (width, height), "#10202b")
                        draw = ImageDraw.Draw(canvas)
                        draw.text((10, 8), f"All 50 frozen final scenes | page {page_index + 1}/5 | "
                                  "exact saved actions | 10 fps", fill="white")
                        for index, (episode, env) in enumerate(zip(group, environments, strict=True)):
                            trace = episode["trace"]
                            if 0 < frame_step <= len(trace):
                                row = trace[frame_step - 1]
                                _, _, _, _, info = env.step(np.asarray(row["action"], dtype=np.float32))
                                page_error = max(page_error, abs(float(info["coverage"]) - row["coverage"]))
                                images[index] = Image.fromarray(np.asarray(env.render())).convert("RGB")
                            if page_error > 1e-7:
                                raise ValueError("Montage replay diverged from saved coverage")
                            shown_step = min(frame_step, len(trace))
                            coverage = trace[shown_step - 1]["coverage"] if shown_step else episode["initial_coverage"]
                            x, y = (index % 5) * 192, 32 + (index // 5) * tile_height
                            canvas.paste(images[index].resize((192, 192), Image.Resampling.LANCZOS), (x, y))
                            metric = episode_metrics(episode)
                            draw.text((x + 5, y + 197), episode["scene_id"], fill="white")
                            done = " FINISHED" if frame_step >= len(trace) else ""
                            draw.text((x + 5, y + 214), f"step {shown_step}/{len(trace)} cov {coverage:.1%}", fill="white")
                            draw.text((x + 5, y + 231), f"87:{'Y' if metric['success_87'] else 'N'} "
                                      f"95:{'Y' if metric['success_95'] else 'N'}{done}", fill="#9ad5ce")
                        video_frame = av.VideoFrame.from_image(canvas)
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                        total_frames += 1
                    pages.append({"page": page_index + 1, "scene_ids": [ep["scene_id"] for ep in group],
                                  "frames": longest + 1, "max_coverage_error": page_error})
                    print(f"Montage page {page_index + 1}/5 verified: {longest + 1} frames, "
                          f"max coverage error={page_error:.3g}", flush=True)
                    maximum_error = max(maximum_error, page_error)
                finally:
                    for env in environments:
                        env.close()
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    proof = {"source_results": str(results_path.resolve()), "source_sha256": source_hash,
             "video_sha256": file_hash(destination), "scene_ids": identifiers, "scene_count": 50,
             "frames": total_frames, "fps": 10, "duration_seconds": total_frames / 10,
             "size": [width, height], "pages": pages, "max_coverage_error": maximum_error,
             "method": "all 50 scenes in frozen order; exact physical-action replay; 5x2 tiles per page; "
                       "finished scenes hold their last frame; no outcome-based omissions"}
    atomic_json(proof_path, proof)
    return proof


def charts(summary: dict, rows: list[dict], output: Path, log: Path | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    rates = [summary[f"success_rate_{t}"] for t in (87, 95)]
    intervals = [summary[f"wilson_95_{t}"] for t in (87, 95)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1], rates, width=.5, color=["#276E8F", "#61A5A0"])
    ax.errorbar([0, 1], rates,
                yerr=[[rates[i] - intervals[i][0] for i in range(2)],
                      [intervals[i][1] - rates[i] for i in range(2)]],
                fmt="none", color="#183642", capsize=6)
    ax.set(xticks=[0, 1], xticklabels=["Coverage > 87%", "Coverage > 95%"], ylim=(0, 1.13),
           ylabel="Success rate", title="Final 50 scenes · 95% Wilson intervals")
    for i, t in enumerate((87, 95)):
        ax.text(i, 1.04, f"{summary[f'success_count_{t}']}/50 = {rates[i]:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "success-rates.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([r["max_coverage"] for r in rows], bins=np.linspace(0, 1, 11), alpha=.7,
            color="#276E8F", label="Maximum coverage")
    ax.hist([r["final_coverage"] for r in rows], bins=np.linspace(0, 1, 11), histtype="step",
            color="#C3733D", linewidth=2, label="Final coverage")
    for threshold in (.87, .95):
        ax.axvline(threshold, color="#515C66", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), xlabel="Coverage", ylabel="Scene count", title="Final 50 · coverage distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "coverage-distribution.png")
    plt.close(fig)
    meta = {"success_rates": "success-rates.png", "coverage_distribution": "coverage-distribution.png"}
    if log is not None:
        pairs = re.findall(r"训练\s+(\d+)/\d+\s*\|\s*当前 loss\s+([\d.eE+-]+)", log.read_text())
        values = {int(step): float(loss) for step, loss in pairs}
        if not values or not all(math.isfinite(x) for x in values.values()):
            raise ValueError("Training log has no finite recognized loss records")
        steps = sorted(values)
        losses = np.array([values[s] for s in steps])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, losses, alpha=.25, linewidth=.5, color="#276E8F", label="Logged batch loss")
        window = min(100, len(losses))
        means = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], means, color="#183642", label=f"Last {window} logged records mean")
        ax.set(xlabel="Training update", ylabel="Noise prediction loss", title="Training diagnostics · not policy success")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "training-loss.png")
        plt.close(fig)
        meta["training_loss"] = {"path": "training-loss.png", "source_sha256": file_hash(log),
                                 "records": len(steps), "loss_values": "rounded console values"}
    return meta


def build(args) -> dict:
    result, pool, selection, rows = validate_inputs(args.results, args.scenes, args.selection)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    choices = choose_videos(rows, selection)
    # 先持久化预声明规则选出的ID；不能根据视频观感重新挑选。
    atomic_json(output / "video-selection.json", {"source_sha256": file_hash(args.results),
                "rule": selection["video_example_rule"], "choices": choices}, frozen=True)
    summary = recompute(result)
    video_artifacts = videos(result, args.results, choices, args.video_dir)
    montage_proof = montage(result, args.results, args.video_dir)
    figure_artifacts = charts(summary, rows, output, args.training_log)
    with (output / "per-scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = {}
    for label, path in (("results", args.results), ("scenes", args.scenes), ("selection", args.selection),
                        ("manifest", args.results.parent / "manifest.json")):
        target = output / "evidence" / f"{label}.json"
        copy_evidence(path, target)
        sources[label] = {"path": str(path.resolve()), "sha256": file_hash(path),
                          "report_copy": str(target.relative_to(output))}
    initial = {str(t): sum(r["initial_coverage"] > t / 100 for r in rows) for t in (87, 95)}
    limits = [
        "全部 206 条演示用于训练；仅一个训练随机种子（seed=0），没有跨训练种子稳定性证据。",
        "最终场景在开发集选模锁定后独立随机生成，并排除记录在案的开发/debug种子及初始状态；无法证明与训练姿态完全不重合。",
        "Wilson 95% 区间反映这 50 个场景的抽样不确定性；不是未来成功率保证，也不是跨训练种子的误差条。",
        "两项成功均按整条轨迹（包括初始状态）的最大 coverage 严格大于阈值计算；不是终态覆盖率成功率。",
        "达到 >95% 或环境终止则提前结束，否则执行最多 300 步；达到 >87% 后仍继续运行。",
        "loss 只作为噪声预测训练诊断，不能据此断言闭环策略成功；本报告结论来自最终闭环 trace。",
        "两个示例视频按预声明规则选取，只用于展示；另提供包含全部 50 场的固定顺序拼接视频，短轨迹结束后停帧标记。",
        "该报告核验冻结选模记录中的排序和哈希关联；未重新运行开发评测或训练。",
    ]
    report = {"schema": "pusht_acceptance_v1", "evidence_kind": "formal", "summary": summary,
              "initial_success_counts": initial, "selected_step": selection["chosen_step"],
              "checkpoint_sha256": selection["chosen_sha256"], "scenes_hash": pool["scenes_hash"],
              "protocol": pool["protocol"], "protocol_amendment": AMENDMENT,
              "training": {"episodes": 206, "training_seeds": [0], "weights": "ema"},
              "runtime": result["runner"]["runtime"], "sources": sources,
              "figures": figure_artifacts, "videos": video_artifacts, "montage": montage_proof, "limits": limits}
    atomic_json(output / "acceptance.json", report)
    lines = ["# Push-T Diffusion Policy 最终验收", "",
             f"已核验最终 **50/50** 个冻结场景。采用开发集选出的 **step {selection['chosen_step']} EMA** 权重。", "",
             "| 指标 | 成功数 / 场景数 | 成功率 | Wilson 95% 区间 |", "|---|---:|---:|---:|"]
    for t in (87, 95):
        lo, hi = summary[f"wilson_95_{t}"]
        lines.append(f"| 最大覆盖率 >{t}% | {summary[f'success_count_{t}']}/50 | "
                     f"{summary[f'success_rate_{t}']:.1%} | {lo:.1%}–{hi:.1%} |")
    lines += ["", f"平均最大覆盖率：**{summary['mean_max_coverage']:.2%}**；"
              f"平均终态覆盖率：**{summary['mean_final_coverage']:.2%}**。",
              f"规划延迟 p50 / p95：{summary['inference_p50_ms']} / {summary['inference_p95_ms']} ms。",
              f"初始状态已 >87%：{initial['87']}/50；初始已 >95%：{initial['95']}/50。", "",
              "![成功率与区间](success-rates.png)", "", "![覆盖率分布](coverage-distribution.png)", "",
              "## 协议与选模", "",
              "U-Net；EMA；DDIM 20 步；每次预测 16 个动作、执行前 4 个；相邻两帧观测逐环境步更新；最多 300 步。",
              "先在相同的 50 个开发场景比较 5k–40k 共 8 个快照，按 >87% 成功数、平均最大覆盖率、较早训练步数依次选模。",
              "最终评测按用户要求采用独立 50 场景，使用显式 v2 协议修订；不将原 v1 的 100 场景要求伪称为已满足。", "",
              f"检查点 SHA256：`{selection['chosen_sha256']}`。",
              f"最终场景 SHA256：`{pool['scenes_hash']}`。", "", "## 固定规则回放视频", ""]
    for category, label in (("success", "最小 scene_id 的 >95% 成功场景"), ("failure", "最小 scene_id 的 >87% 失败场景")):
        artifact = video_artifacts[category]
        if artifact["status"] == "category_absent":
            lines.append(f"{label}：本批不存在此类别，未制造或替换示例。")
        else:
            video = Path(artifact["video"])
            rel_video = Path(os.path.relpath(video, output)).as_posix()
            rel_gif = Path(os.path.relpath(artifact["gif"], output)).as_posix()
            lines += [f"{label}：`{artifact['scene_id']}`。[完整视频]({rel_video})；回放最大 coverage 误差 "
                      f"{artifact['provenance']['max_coverage_error']:.3g}。", "", f"![{label}]({rel_gif})", ""]
    montage_relative = Path(os.path.relpath(args.video_dir / "all50.mp4", output)).as_posix()
    lines += [f"[全部 50 场固定顺序视频]({montage_relative})：5 页，每页 10 场；"
              f"{montage_proof['duration_seconds']:.1f} 秒；逐步回放最大 coverage 误差 "
              f"{montage_proof['max_coverage_error']:.3g}。", ""]
    if args.training_log:
        lines += ["## 训练诊断", "", "![训练损失](training-loss.png)", "",
                  "曲线来自日志中已舍入的 loss，仅作训练诊断。", ""]
    lines += ["## 限制", "", *[f"- {line}" for line in limits], "", "## 原始证据", "",
              "[逐场 CSV](per-scene.csv) · [结构化报告](acceptance.json) · [原始结果](evidence/results.json) · "
              "[冻结场景](evidence/scenes.json) · [选模锁](evidence/selection.json) · [运行清单](evidence/manifest.json)", ""]
    (output / "acceptance.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "scenes", "selection", "output-dir", "video-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"report": str(args.output_dir / "acceptance.md"), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
