# sad talker와 latentsync 모델을 다운로드하고 환경을 설정하는 스크립트
# 실행 명령어: bash scripts/bootstrap_models.sh "$PWD/models"

#!/usr/bin/env bash
set -euo pipefail

# Linux GPU 서버에서 실행한다. 첫 번째 인자를 생략하면 프로젝트의 models/를 사용한다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-${TALKING_HEAD_MODELS_ROOT:-${PROJECT_ROOT}/models}}"
SADTALKER_REF="${SADTALKER_REF:-main}"
LATENTSYNC_REF="${LATENTSYNC_REF:-main}"

SADTALKER_REPO="${MODEL_ROOT}/SadTalker"
LATENTSYNC_REPO="${MODEL_ROOT}/LatentSync"
SADTALKER_ENV="${MODEL_ROOT}/envs/sadtalker"
LATENTSYNC_ENV="${MODEL_ROOT}/envs/latentsync"

for executable in git conda wget; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "필수 실행 파일이 없습니다: ${executable}" >&2
        exit 1
    fi
done

mkdir -p "${MODEL_ROOT}" "${MODEL_ROOT}/envs"

clone_if_missing() {
    local url="$1"
    local ref="$2"
    local destination="$3"

    if [[ -d "${destination}/.git" ]]; then
        echo "기존 저장소를 사용합니다: ${destination}"
        return
    fi
    if [[ -e "${destination}" ]]; then
        echo "대상 경로가 Git 저장소가 아닙니다: ${destination}" >&2
        exit 1
    fi

    git clone "${url}" "${destination}"
    git -C "${destination}" checkout "${ref}"
}

clone_if_missing \
    "https://github.com/OpenTalker/SadTalker.git" \
    "${SADTALKER_REF}" \
    "${SADTALKER_REPO}"
clone_if_missing \
    "https://github.com/bytedance/LatentSync.git" \
    "${LATENTSYNC_REF}" \
    "${LATENTSYNC_REPO}"

if [[ ! -x "${SADTALKER_ENV}/bin/python" ]]; then
    conda create -y -p "${SADTALKER_ENV}" python=3.8 ffmpeg
    conda run -p "${SADTALKER_ENV}" python -m pip install \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113
    conda run -p "${SADTALKER_ENV}" python -m pip install \
        -r "${SADTALKER_REPO}/requirements.txt"
else
    echo "기존 SadTalker 환경을 사용합니다: ${SADTALKER_ENV}"
fi

download_if_missing() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.download"

    if [[ -s "${destination}" ]]; then
        echo "기존 체크포인트를 사용합니다: ${destination}"
        return
    fi
    mkdir -p "$(dirname "${destination}")"
    wget --progress=dot:giga "${url}" -O "${temporary}"
    mv "${temporary}" "${destination}"
}

# SadTalker 공식 다운로드 스크립트에 등록된 파일을 개별적으로 받는다.
# 다운로드가 중단되더라도 완료된 파일은 재사용하고 누락된 파일만 다시 받는다.
download_if_missing \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" \
    "${SADTALKER_REPO}/checkpoints/mapping_00109-model.pth.tar"
download_if_missing \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" \
    "${SADTALKER_REPO}/checkpoints/mapping_00229-model.pth.tar"
download_if_missing \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" \
    "${SADTALKER_REPO}/checkpoints/SadTalker_V0.0.2_256.safetensors"
download_if_missing \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors" \
    "${SADTALKER_REPO}/checkpoints/SadTalker_V0.0.2_512.safetensors"
download_if_missing \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" \
    "${SADTALKER_REPO}/gfpgan/weights/alignment_WFLW_4HG.pth"
download_if_missing \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
    "${SADTALKER_REPO}/gfpgan/weights/detection_Resnet50_Final.pth"
download_if_missing \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    "${SADTALKER_REPO}/gfpgan/weights/GFPGANv1.4.pth"
download_if_missing \
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
    "${SADTALKER_REPO}/gfpgan/weights/parsing_parsenet.pth"

if [[ ! -x "${LATENTSYNC_ENV}/bin/python" ]]; then
    conda create -y -p "${LATENTSYNC_ENV}" python=3.10.13 ffmpeg
    conda run -p "${LATENTSYNC_ENV}" python -m pip install \
        -r "${LATENTSYNC_REPO}/requirements.txt"
else
    echo "기존 LatentSync 환경을 사용합니다: ${LATENTSYNC_ENV}"
fi

if [[ ! -f "${LATENTSYNC_REPO}/checkpoints/latentsync_unet.pt" ]] || \
   [[ ! -f "${LATENTSYNC_REPO}/checkpoints/whisper/tiny.pt" ]]; then
    conda run -p "${LATENTSYNC_ENV}" python -c \
        "import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/LatentSync-1.6', local_dir=sys.argv[1], allow_patterns=['latentsync_unet.pt', 'whisper/tiny.pt'])" \
        "${LATENTSYNC_REPO}/checkpoints"
else
    echo "기존 LatentSync 체크포인트를 사용합니다: ${LATENTSYNC_REPO}/checkpoints"
fi

echo
echo "모델 준비가 완료되었습니다."
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "파이프라인 실행 예시:"
echo "python ${PROJECT_ROOT}/talking_head_pipeline.py --source-image FACE.jpg --audio VOICE.wav --output final.mp4 --models-root ${MODEL_ROOT}"
