#!/usr/bin/env python3
"""SadTalker와 LatentSync를 분리된 순차 파이프라인으로 실행한다.

이 파일은 의도적으로 Python 표준 라이브러리에만 의존한다. 각 모델을
서로 다른 환경의 Python으로 실행하므로 호환되지 않는 모델 의존성을
한 환경에 함께 설치할 필요가 없다.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger("talking-head-pipeline")


class PipelineError(RuntimeError):
    """파이프라인 단계가 올바른 결과를 만들지 못했을 때 발생하는 예외다."""


@dataclass(frozen=True)
class RuntimePaths:
    models_root: Path
    sadtalker_repo: Path
    latentsync_repo: Path
    sadtalker_python: Path
    latentsync_python: Path
    ffmpeg: Path | str


@dataclass(frozen=True)
class PipelineOptions:
    source_image: Path
    audio: Path
    output: Path
    work_root: Path
    pose_style: int
    expression_scale: float
    size: int
    preprocess: str
    still: bool
    enhancer: str | None
    inference_steps: int
    guidance_scale: float
    seed: int
    deepcache: bool
    keep_workdir: bool
    overwrite: bool


def _existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _inference_steps(value: str) -> int:
    parsed = int(value)
    if not 20 <= parsed <= 50:
        raise argparse.ArgumentTypeError("inference steps must be between 20 and 50")
    return parsed


def _pose_style(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 46:
        raise argparse.ArgumentTypeError("pose style must be between 0 and 45")
    return parsed


def _guidance_scale(value: str) -> float:
    parsed = float(value)
    if not 1.0 <= parsed <= 3.0:
        raise argparse.ArgumentTypeError("guidance scale must be between 1.0 and 3.0")
    return parsed


def _resolve_environment_python(environment: Path) -> Path:
    candidates = (
        environment / "bin" / "python",
        environment / "Scripts" / "python.exe",
        environment / "python.exe",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _resolve_ffmpeg(models_root: Path, override: str | None) -> Path | str:
    if override:
        candidate = Path(override).expanduser()
        if candidate.parent != Path(".") or candidate.is_absolute():
            return candidate.resolve()
        return override

    environment = models_root / "envs" / "latentsync"
    candidates = (
        environment / "bin" / "ffmpeg",
        environment / "Library" / "bin" / "ffmpeg.exe",
    )
    return next((path for path in candidates if path.is_file()), "ffmpeg")


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    models_root = Path(args.models_root).expanduser().resolve()
    sadtalker_repo = (
        Path(args.sadtalker_repo).expanduser().resolve()
        if args.sadtalker_repo
        else models_root / "SadTalker"
    )
    latentsync_repo = (
        Path(args.latentsync_repo).expanduser().resolve()
        if args.latentsync_repo
        else models_root / "LatentSync"
    )
    sadtalker_python = (
        Path(args.sadtalker_python).expanduser().resolve()
        if args.sadtalker_python
        else _resolve_environment_python(models_root / "envs" / "sadtalker")
    )
    latentsync_python = (
        Path(args.latentsync_python).expanduser().resolve()
        if args.latentsync_python
        else _resolve_environment_python(models_root / "envs" / "latentsync")
    )
    return RuntimePaths(
        models_root=models_root,
        sadtalker_repo=sadtalker_repo,
        latentsync_repo=latentsync_repo,
        sadtalker_python=sadtalker_python,
        latentsync_python=latentsync_python,
        ffmpeg=_resolve_ffmpeg(models_root, args.ffmpeg),
    )


def validate_runtime(runtime: RuntimePaths, options: PipelineOptions) -> None:
    mapping_checkpoint = (
        "mapping_00109-model.pth.tar"
        if "full" in options.preprocess
        else "mapping_00229-model.pth.tar"
    )
    required = {
        "SadTalker inference script": runtime.sadtalker_repo / "inference.py",
        f"SadTalker {options.size} checkpoint": runtime.sadtalker_repo
        / "checkpoints"
        / f"SadTalker_V0.0.2_{options.size}.safetensors",
        "SadTalker mapping checkpoint": runtime.sadtalker_repo
        / "checkpoints"
        / mapping_checkpoint,
        "LatentSync inference script": runtime.latentsync_repo / "scripts" / "inference.py",
        "LatentSync config": runtime.latentsync_repo / "configs" / "unet" / "stage2_512.yaml",
        "LatentSync checkpoint": runtime.latentsync_repo / "checkpoints" / "latentsync_unet.pt",
        "LatentSync Whisper checkpoint": runtime.latentsync_repo / "checkpoints" / "whisper" / "tiny.pt",
        "SadTalker Python": runtime.sadtalker_python,
        "LatentSync Python": runtime.latentsync_python,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]

    if isinstance(runtime.ffmpeg, Path):
        if not runtime.ffmpeg.is_file():
            missing.append(f"FFmpeg: {runtime.ffmpeg}")
    elif shutil.which(runtime.ffmpeg) is None:
        missing.append(f"FFmpeg executable on PATH: {runtime.ffmpeg}")

    if missing:
        details = "\n  - ".join(missing)
        raise PipelineError(
            "model runtime is incomplete. Check the remote model repositories, "
            f"checkpoints, and Python paths below:\n  - {details}"
        )


def run_command(command: Sequence[str | Path], *, cwd: Path | None = None) -> None:
    rendered = [str(part) for part in command]
    LOGGER.info("Running: %s", shlex.join(rendered))
    try:
        subprocess.run(rendered, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(f"executable was not found: {rendered[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            f"command failed with exit code {exc.returncode}: {shlex.join(rendered)}"
        ) from exc


def normalize_audio(ffmpeg: Path | str, source: Path, destination: Path) -> None:
    run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source,
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            destination,
        ]
    )
    require_nonempty_file(destination, "normalized audio")


def run_sadtalker(
    runtime: RuntimePaths,
    options: PipelineOptions,
    normalized_audio: Path,
    result_dir: Path,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=False)
    command: list[str | Path] = [
        runtime.sadtalker_python,
        runtime.sadtalker_repo / "inference.py",
        "--source_image",
        options.source_image,
        "--driven_audio",
        normalized_audio,
        "--result_dir",
        result_dir,
        "--checkpoint_dir",
        runtime.sadtalker_repo / "checkpoints",
        "--pose_style",
        str(options.pose_style),
        "--expression_scale",
        str(options.expression_scale),
        "--size",
        str(options.size),
        "--preprocess",
        options.preprocess,
    ]
    if options.still:
        command.append("--still")
    if options.enhancer:
        command.extend(("--enhancer", options.enhancer))

    run_command(command, cwd=runtime.sadtalker_repo)
    candidates = sorted(
        (path for path in result_dir.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise PipelineError(f"SadTalker did not create an MP4 under {result_dir}")
    if len(candidates) > 1:
        LOGGER.warning("SadTalker created multiple MP4 files; using newest: %s", candidates[0])
    return candidates[0]


def normalize_video(ffmpeg: Path | str, source: Path, destination: Path) -> None:
    run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=25",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            destination,
        ]
    )
    require_nonempty_file(destination, "25 FPS intermediate video")


def run_latentsync(
    runtime: RuntimePaths,
    options: PipelineOptions,
    video: Path,
    normalized_audio: Path,
    destination: Path,
    temp_dir: Path,
) -> None:
    # LatentSync 1.6은 현재 마지막 FFmpeg 명령을 shell=True로 실행하면서
    # temp_dir와 video_out_path를 따옴표로 감싸지 않는다. 공식 구현이 바뀔
    # 때까지 생성 경로에 보수적인 문자 집합만 허용한다.
    for path in (destination, temp_dir):
        if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", str(path)) is None:
            raise PipelineError(
                "the work path contains spaces or shell metacharacters that the "
                f"upstream LatentSync FFmpeg call cannot safely handle: {path}"
            )

    command: list[str | Path] = [
        runtime.latentsync_python,
        "-m",
        "scripts.inference",
        "--unet_config_path",
        runtime.latentsync_repo / "configs" / "unet" / "stage2_512.yaml",
        "--inference_ckpt_path",
        runtime.latentsync_repo / "checkpoints" / "latentsync_unet.pt",
        "--inference_steps",
        str(options.inference_steps),
        "--guidance_scale",
        str(options.guidance_scale),
        "--video_path",
        video,
        "--audio_path",
        normalized_audio,
        "--video_out_path",
        destination,
        "--temp_dir",
        temp_dir,
        "--seed",
        str(options.seed),
    ]
    if options.deepcache:
        command.append("--enable_deepcache")

    run_command(command, cwd=runtime.latentsync_repo)
    require_nonempty_file(destination, "LatentSync output")


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PipelineError(f"{label} was not created or is empty: {path}")


def publish_output(source: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise PipelineError(f"output already exists (use --overwrite): {destination}")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def execute_pipeline(runtime: RuntimePaths, options: PipelineOptions) -> Path:
    validate_runtime(runtime, options)
    if options.output.exists() and not options.overwrite:
        raise PipelineError(f"output already exists (use --overwrite): {options.output}")

    run_dir = options.work_root / f"run-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    succeeded = False
    LOGGER.info("Work directory: %s", run_dir)

    try:
        normalized_audio = run_dir / "audio_16k_mono.wav"
        normalize_audio(runtime.ffmpeg, options.audio, normalized_audio)

        sad_video = run_sadtalker(
            runtime,
            options,
            normalized_audio,
            run_dir / "sadtalker",
        )
        LOGGER.info("SadTalker output: %s", sad_video)

        normalized_video = run_dir / "sadtalker_25fps.mp4"
        normalize_video(runtime.ffmpeg, sad_video, normalized_video)

        latent_output = run_dir / "latentsync_output.mp4"
        run_latentsync(
            runtime,
            options,
            normalized_video,
            normalized_audio,
            latent_output,
            run_dir / "latentsync-temp",
        )
        publish_output(latent_output, options.output, overwrite=options.overwrite)
        succeeded = True
        return options.output
    finally:
        if succeeded and not options.keep_workdir:
            shutil.rmtree(run_dir)
        else:
            LOGGER.info("Intermediate files retained at: %s", run_dir)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_models_root = Path(
        os.environ.get("TALKING_HEAD_MODELS_ROOT", project_root / "models")
    ).expanduser()

    parser = argparse.ArgumentParser(
        description="Generate a talking-head video with SadTalker, then refine lips with LatentSync 1.6."
    )
    parser.add_argument("--source-image", required=True, type=_existing_path)
    parser.add_argument("--audio", required=True, type=_existing_path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models-root", default=default_models_root)
    parser.add_argument(
        "--sadtalker-repo",
        help="Override the repository path (default: <models-root>/SadTalker)",
    )
    parser.add_argument(
        "--latentsync-repo",
        help="Override the repository path (default: <models-root>/LatentSync)",
    )
    parser.add_argument("--sadtalker-python", help="Override the SadTalker environment Python path")
    parser.add_argument("--latentsync-python", help="Override the LatentSync environment Python path")
    parser.add_argument("--ffmpeg", help="Override the FFmpeg executable path")
    parser.add_argument("--work-root", type=Path, default=project_root / "work")

    sad = parser.add_argument_group("SadTalker")
    sad.add_argument("--pose-style", type=_pose_style, default=0)
    sad.add_argument("--expression-scale", type=float, default=1.0)
    sad.add_argument("--size", type=int, choices=(256, 512), default=512)
    sad.add_argument(
        "--preprocess",
        choices=("crop", "extcrop", "resize", "full", "extfull"),
        default="crop",
    )
    sad.add_argument("--still", action="store_true")
    sad.add_argument("--enhancer", choices=("gfpgan", "RestoreFormer"))

    latent = parser.add_argument_group("LatentSync")
    latent.add_argument("--inference-steps", type=_inference_steps, default=20)
    latent.add_argument("--guidance-scale", type=_guidance_scale, default=1.5)
    latent.add_argument("--seed", type=int, default=1247)
    latent.add_argument(
        "--no-deepcache",
        action="store_false",
        dest="deepcache",
        help="Disable LatentSync DeepCache (enabled by default)",
    )
    latent.set_defaults(deepcache=True)

    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        source_image=args.source_image.resolve(),
        audio=args.audio.resolve(),
        output=args.output.expanduser().resolve(),
        work_root=args.work_root.expanduser().resolve(),
        pose_style=args.pose_style,
        expression_scale=args.expression_scale,
        size=args.size,
        preprocess=args.preprocess,
        still=args.still,
        enhancer=args.enhancer,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        deepcache=args.deepcache,
        keep_workdir=args.keep_workdir,
        overwrite=args.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        output = execute_pipeline(resolve_runtime_paths(args), options_from_args(args))
    except (OSError, PipelineError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Final video: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
