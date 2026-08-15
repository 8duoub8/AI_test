#!/usr/bin/env bash
set -euo pipefail

# SadTalker와 LatentSync 모델을 다운로드하고 환경을 설정한다.
# 실행 명령어: bash scripts/bootstrap_models.sh "$PWD/models"

# Linux GPU 서버에서 실행한다. 첫 번째 인자를 생략하면 프로젝트의 models/를 사용한다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-${TALKING_HEAD_MODELS_ROOT:-${PROJECT_ROOT}/models}}"
SADTALKER_REF="${SADTALKER_REF:-main}"
LATENTSYNC_REF="${LATENTSYNC_REF:-main}"

SADTALKER_REPO="${MODEL_ROOT}/SadTalker"
LATENTSYNC_REPO="${MODEL_ROOT}/LatentSync"
SADTALKER_ENV="${MODEL_ROOT}/envs/sadtalker"
LATENTSYNC_ENV="${MODEL_ROOT}/envs/latentsync"
MAMBA_ROOT_PREFIX="${MODEL_ROOT}/.micromamba-root"
export MAMBA_ROOT_PREFIX

for executable in git wget tar; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "필수 실행 파일이 없습니다: ${executable}" >&2
        exit 1
    fi
done

mkdir -p "${MODEL_ROOT}" "${MODEL_ROOT}/envs"

find_ca_bundle() {
    local candidate
    local variable_name

    for variable_name in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
        candidate="$(printenv "${variable_name}" 2>/dev/null || true)"
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    if command -v python >/dev/null 2>&1; then
        candidate="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    fi

    for candidate in \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/pki/tls/certs/ca-bundle.crt \
        /etc/ssl/ca-bundle.pem; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
}

CA_BUNDLE="$(find_ca_bundle || true)"
WGET_TLS_OPTIONS=()
if [[ -n "${CA_BUNDLE}" ]]; then
    echo "TLS 인증서 묶음을 사용합니다: ${CA_BUNDLE}"
    WGET_TLS_OPTIONS=(--ca-certificate="${CA_BUNDLE}")
    export SSL_CERT_FILE="${CA_BUNDLE}"
    export REQUESTS_CA_BUNDLE="${CA_BUNDLE}"
    export CURL_CA_BUNDLE="${CA_BUNDLE}"
    export GIT_SSL_CAINFO="${CA_BUNDLE}"
    export PIP_CERT="${CA_BUNDLE}"
else
    echo "사용 가능한 TLS 인증서 묶음을 찾지 못했습니다." >&2
    echo "현재 Python 환경에 certifi를 설치한 뒤 다시 실행하십시오." >&2
    exit 1
fi

install_micromamba() {
    local platform
    local tools_dir="${MODEL_ROOT}/.tools"
    local archive="${tools_dir}/micromamba.tar.bz2"
    local downloaded="${archive}.download"

    case "$(uname -m)" in
        x86_64) platform="linux-64" ;;
        aarch64|arm64) platform="linux-aarch64" ;;
        ppc64le) platform="linux-ppc64le" ;;
        *)
            echo "지원하지 않는 CPU 아키텍처입니다: $(uname -m)" >&2
            exit 1
            ;;
    esac

    mkdir -p "${tools_dir}"
    echo "Conda 계열 명령이 없어 Micromamba를 내려받습니다."
    wget "${WGET_TLS_OPTIONS[@]}" --progress=dot:giga \
        "https://micro.mamba.pm/api/micromamba/${platform}/latest" \
        -O "${downloaded}"
    mv "${downloaded}" "${archive}"
    tar -xjf "${archive}" -C "${tools_dir}" bin/micromamba
    mv "${tools_dir}/bin/micromamba" "${tools_dir}/micromamba"
    rmdir "${tools_dir}/bin"
    rm -f "${archive}"
    chmod +x "${tools_dir}/micromamba"
    ENV_MANAGER="${tools_dir}/micromamba"
    ENV_MANAGER_KIND="micromamba"
}

if command -v conda >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v conda)"
    ENV_MANAGER_KIND="conda"
elif command -v micromamba >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v micromamba)"
    ENV_MANAGER_KIND="micromamba"
elif [[ -x "${MODEL_ROOT}/.tools/micromamba" ]]; then
    ENV_MANAGER="${MODEL_ROOT}/.tools/micromamba"
    ENV_MANAGER_KIND="micromamba"
else
    install_micromamba
fi

echo "환경 관리자: ${ENV_MANAGER_KIND} (${ENV_MANAGER})"

create_environment() {
    local prefix="$1"
    shift
    "${ENV_MANAGER}" create -y \
        --override-channels \
        -c conda-forge \
        -p "${prefix}" \
        "$@"
}

run_in_environment() {
    local prefix="$1"
    shift
    "${ENV_MANAGER}" run -p "${prefix}" "$@"
}

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
    create_environment "${SADTALKER_ENV}" python=3.8 ffmpeg pip
    run_in_environment "${SADTALKER_ENV}" python -m pip install \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113
    run_in_environment "${SADTALKER_ENV}" python -m pip install \
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
    wget "${WGET_TLS_OPTIONS[@]}" --progress=dot:giga "${url}" -O "${temporary}"
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
    create_environment "${LATENTSYNC_ENV}" python=3.10.13 ffmpeg pip
    run_in_environment "${LATENTSYNC_ENV}" python -m pip install \
        -r "${LATENTSYNC_REPO}/requirements.txt"
else
    echo "기존 LatentSync 환경을 사용합니다: ${LATENTSYNC_ENV}"
fi

if [[ ! -f "${LATENTSYNC_REPO}/checkpoints/latentsync_unet.pt" ]] || \
   [[ ! -f "${LATENTSYNC_REPO}/checkpoints/whisper/tiny.pt" ]]; then
    run_in_environment "${LATENTSYNC_ENV}" python -c \
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
