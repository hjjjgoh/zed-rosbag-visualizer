#!/bin/bash
# GNU Parallel을 사용한 병렬 실행 스크립트
# 설치: sudo apt-get install parallel (Ubuntu/Debian)
# 설치: brew install parallel (Mac)

# 로그 디렉토리 생성
mkdir -p logs

# 실험 설정들을 정의
# 형식: "bag_path,fps,algorithm,output_dir"
experiments=(
    "data/rosbag2_4,5,direct,output/exp_fps5_direct"
    "data/rosbag2_4,10,direct,output/exp_fps10_direct"
    "data/rosbag2_4,15,direct,output/exp_fps15_direct"
    "data/rosbag2_4,5,batch,output/exp_fps5_batch"
    "data/rosbag2_4,10,batch,output/exp_fps10_batch"
    "data/rosbag2_4,15,batch,output/exp_fps15_batch"
)

# 병렬 실행 함수
run_experiment() {
    IFS=',' read -r bag_path fps algorithm output_dir <<< "$1"
    
    echo "Starting: FPS=$fps, Algorithm=$algorithm"
    
    uv run python scripts_edit/preprocess_edit.py "$bag_path" \
        --fps "$fps" \
        --depth-source foundation \
        --matching-algorithm "$algorithm" \
        --output-dir "$output_dir"
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: FPS=$fps, Algorithm=$algorithm"
    else
        echo "✗ Failed: FPS=$fps, Algorithm=$algorithm (exit code: $exit_code)"
    fi
    
    return $exit_code
}

export -f run_experiment

# GNU parallel을 사용할 수 있는지 확인
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for parallel execution..."
    # -j: 동시 실행 작업 수 (기본값은 CPU 코어 수)
    # --bar: 진행률 표시
    # --joblog: 작업 로그 저장
    printf '%s\n' "${experiments[@]}" | parallel -j 3 --bar --joblog logs/parallel_jobs.log run_experiment
else
    echo "GNU Parallel not found. Falling back to basic parallel execution..."
    
    # 기본 병렬 실행 (GNU parallel 없이)
    pids=()
    for exp in "${experiments[@]}"; do
        run_experiment "$exp" &
        pids+=($!)
    done
    
    # 모든 작업 완료 대기
    for pid in ${pids[@]}; do
        wait $pid
    done
fi

echo "All experiments completed!"

