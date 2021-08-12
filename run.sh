#!/usr/bin/env bash
#privacy-leakage-
python3 examples/cifar10.py \
    --is_log ${IS_LOG} \
    --log_path "log/privacy-leakage-10-isDP=${IS_DP}-type=${DP_TYPE}-pb=${PB}-clipbound=${CLIP_BOUND}-log" \
    --is_dp ${IS_DP} \
    --dp_type ${DP_TYPE} \
    --pb ${PB} \
    --clip_bound ${CLIP_BOUND} \

