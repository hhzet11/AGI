#!/bin/bash

echo "=== DeepSpeed FusedAdam 완전 해결 ==="

# 1. 모든 캐시 완전 삭제
echo "모든 캐시 삭제 중..."
rm -rf ~/.cache/torch_extensions/
rm -rf /tmp/deepspeed_ops/
rm -rf ~/.deepspeed_env/

# 2. FusedAdam 완전 차단 환경변수
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0 
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_UTILS=0
export CUDA_LAUNCH_BLOCKING=1

# 3. PyTorch 기본 optimizer 강제 사용
export USE_FUSED_ADAM=0
export DISABLE_FUSED_ADAM=1

echo "환경 변수 설정 완료!"
echo "DS_BUILD_FUSED_ADAM=$DS_BUILD_FUSED_ADAM"
echo "DISABLE_FUSED_ADAM=$DISABLE_FUSED_ADAM"

# 4. conda 환경 활성화
source ~/miniconda3/bin/activate dsenv
echo "dsenv 환경 활성화됨"

# 5. DeepSpeed 실행 (더 자세한 로그)
echo "=== DeepSpeed 실행 시작 ==="
deepspeed --num_gpus=6 --master_port=29500 train.py 2>&1 | tee deepspeed_output.log

echo "완료! 로그는 deepspeed_output.log에 저장됨"

