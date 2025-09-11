#!/bin/bash
echo "=== DeepSpeed Eval 시작 ==="

# 캐시 삭제
echo "캐시 삭제 중..."
rm -rf ~/.cache/torch_extensions/

# 환경 변수 설정 (FusedAdam 관련 비활성화)
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_UTILS=0

echo "환경 변수 설정 완료:"
echo "DS_SKIP_CUDA_CHECK=$DS_SKIP_CUDA_CHECK"
echo "DS_BUILD_OPS=$DS_BUILD_OPS"

# conda 환경 활성화
source ~/miniconda3/bin/activate dsenv
echo "dsenv 환경 활성화됨"

# DeepSpeed로 eval.py 실행
echo "DeepSpeed 평가 시작..."
deepspeed --num_gpus=6 --master_port=29501 eval.py \
    --checkpoint ./outputs/checkpoint-9 \
    --batch_size 4 \
    --max_new_tokens 8 \
    --max_length 128

echo "완료! 결과는 test_predictions_tokens.csv와 콘솔 출력 확인"
