#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/walnut_data/yqm/NetKD"
DATASET="ISCXVPN2016"
DATA_ROOT="/walnut_data/yqm/Dataset"
LOG_FILE="${PROJECT_ROOT}/checkpoints/training_orchestration.log"
TEACHER_BATCHES=(128 256 512)
STACK_BATCHES=(128 256)
STUDENT_BATCHES=(64 128 256 512)
PY_LAUNCH=("/root/miniconda3/bin/conda" run -p /root/miniconda3 --no-capture-output python)
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
mkdir -p "${PROJECT_ROOT}/checkpoints"
: >"${LOG_FILE}"

log_msg() {
    printf '%s | %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "${LOG_FILE}"
}

run_teacher() {
    local variant="$1"; shift
    local batch_size="$1"; shift
    local out_dir="$1"; shift
    mkdir -p "${out_dir}"
    log_msg "[${variant}] teacher bs${batch_size} -> ${out_dir}"
    (cd "${PROJECT_ROOT}" && CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" "${PY_LAUNCH[@]}" training/train.py \
        --use_real_data --mode train_teachers \
        --dataset "${DATASET}" --dataset_root "${DATA_ROOT}" \
        --batch_size "${batch_size}" --output_dir "${out_dir}" \
        --epochs_teacher 25 --force_retrain_teachers "$@" | tee -a "${LOG_FILE}")
}

run_stacking() {
    local variant="$1"; shift
    local batch_size="$1"; shift
    local out_dir="$1"; shift
    local tag="$1"; shift
    log_msg "[${variant}] stacking bs${batch_size} using ${out_dir}"
    (cd "${PROJECT_ROOT}" && CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" "${PY_LAUNCH[@]}" training/train.py \
        --use_real_data --mode train_stacking \
        --dataset "${DATASET}" --dataset_root "${DATA_ROOT}" \
        --batch_size "${batch_size}" --output_dir "${out_dir}" "$@" | tee -a "${LOG_FILE}")
    mv -f "${out_dir}/stacking_model.pth" "${out_dir}/stacking_${tag}_bs${batch_size}.pth"
}

run_student() {
    local variant="$1"; shift
    local batch_size="$1"; shift
    local out_dir="$1"; shift
    local stack_tag="$1"; shift
    cp -f "${out_dir}/stacking_${stack_tag}_bs256.pth" "${out_dir}/stacking_model.pth"
    log_msg "[${variant}] student bs${batch_size} (stack ${stack_tag})"
    (cd "${PROJECT_ROOT}" && CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" "${PY_LAUNCH[@]}" training/train.py \
        --use_real_data --mode train_student \
        --dataset "${DATASET}" --dataset_root "${DATA_ROOT}" \
        --batch_size "${batch_size}" --output_dir "${out_dir}" "$@" | tee -a "${LOG_FILE}")
    mv -f "${out_dir}/student_sd_mkd.pth" "${out_dir}/student_${stack_tag}_bs${batch_size}.pth"
}

run_variant_pipeline() {
    local variant="$1"
    local -a flags=()
    if [[ "${variant}" == "eca" ]]; then
        flags=(--resnet_use_eca --mbv3_use_eca)
    fi

    for bs in "${TEACHER_BATCHES[@]}"; do
        run_teacher "${variant}" "${bs}" "${PROJECT_ROOT}/checkpoints/${variant}_teachers_bs${bs}" "${flags[@]}"
    done

    local anchor_dir="${PROJECT_ROOT}/checkpoints/${variant}_pipeline"
    mkdir -p "${anchor_dir}"
    cp -f "${PROJECT_ROOT}/checkpoints/${variant}_teachers_bs256"/*teacher.pth "${anchor_dir}/"

    for stack_bs in "${STACK_BATCHES[@]}"; do
        run_stacking "${variant}" "${stack_bs}" "${anchor_dir}" "ref" "${flags[@]}"
    done

    for stu_bs in "${STUDENT_BATCHES[@]}"; do
        run_student "${variant}" "${stu_bs}" "${anchor_dir}" "ref" "${flags[@]}"
    done
}

run_variant_pipeline "eca"
run_variant_pipeline "baseline"

log_msg "All planned trainings dispatched."
