#!/usr/bin/env bash
# Compatibility entrypoint for the GLM BF16 server/consumer sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/run_finetune_perf_test_bf16_ktransformers.sh"
declare -a FORWARDED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase4-steps)
            [[ $# -ge 2 ]] || {
                echo "Missing value for --phase4-steps" >&2
                exit 2
            }
            FORWARDED+=(--steps "$2")
            shift
            ;;
        --gpu-ids)
            [[ $# -ge 2 ]] || {
                echo "Missing value for --gpu-ids" >&2
                exit 2
            }
            FORWARDED+=(--devices "$2")
            shift
            ;;
        --gpu)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
                echo "--gpu requires a non-negative start id" >&2
                exit 2
            }
            start="$2"
            FORWARDED+=(
                --devices
                "${start},$((start + 1)),$((start + 2)),$((start + 3)),$((start + 4)),$((start + 5)),$((start + 6)),$((start + 7))"
            )
            shift
            ;;
        --gpus)
            [[ $# -ge 2 ]] || {
                echo "Missing value for --gpus" >&2
                exit 2
            }
            case "$2" in
                8) FORWARDED+=(--profile server) ;;
                2) FORWARDED+=(--profile consumer) ;;
                *)
                    echo "GLM BF16 benchmark supports --gpus 8 (server) or 2 (consumer)" >&2
                    exit 2
                    ;;
            esac
            shift
            ;;
        --only-phase4)
            ;;
        --skip-phase4)
            echo "--skip-phase4 has no equivalent in the sequence sweep" >&2
            exit 2
            ;;
        *)
            FORWARDED+=("$1")
            ;;
    esac
    shift
done

echo "NOTE: run_full_ft_test_1gpu_bf16.sh is a compatibility name; use --profile server|consumer|both on the canonical sweep." >&2
exec bash "${TARGET}" "${FORWARDED[@]}"
