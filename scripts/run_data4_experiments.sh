#!/bin/bash
# GNU Parallel을 사용하여 rosbag2_4 데이터에 대한 전처리 실험을 병렬로 실행하는 스크립트

# --- 설정 ---
# 동시 실행할 최대 작업 수
PARALLEL_JOBS=3

# --- 실험 정의 ---
# 형식: "bag_path,fps,width,height,depth_source,output_dir"
#
# 실험 그룹 1: FPS 변경 (해상도 1080x720, depth=both 고정)
fps_experiments=(
    "data/rosbag2_4,5,1080,720,both,output/data4_fps/exp_fps_5"
    "data/rosbag2_4,10,1080,720,both,output/data4_fps/exp_fps_10"
    "data/rosbag2_4,15,1080,720,both,output/data4_fps/exp_fps_15"
)

# 실험 그룹 2: 해상도 변경 (FPS=15, depth=both 고정)
# 너비를 2의 거듭제곱(512, 1024, 2048)으로 설정하고, 1080:720 (3:2) 비율에 맞춰 높이를 계산합니다.
resolution_experiments=(
    "data/rosbag2_4,15,512,341,both,output/data4_resolution/exp_res_512x341"
    "data/rosbag2_4,15,1024,682,both,output/data4_resolution/exp_res_1024x682"
    "data/rosbag2_4,15,2048,1365,both,output/data4_resolution/exp_res_2048x1365"
)

# 모든 실험을 하나의 배열로 결합
all_experiments=("${fps_experiments[@]}" "${resolution_experiments[@]}")


# --- 실행 로직 ---

# 로그 및 출력 디렉토리 생성
mkdir -p logs
mkdir -p output/data4_fps
mkdir -p output/data4_resolution

# 병렬 실행 함수
run_experiment() {
    IFS=',' read -r bag_path fps width height depth_source output_dir <<< "$1"
    
    # 출력 디렉토리 이름에서 로그 파일 이름 생성
    log_file_name=$(basename "$output_dir")
    
    echo "Starting: Bag=$bag_path, FPS=$fps, Res=${width}x${height}, Depth=$depth_source"
    
    # stdout과 stderr를 모두 로그 파일로 리디렉션
    uv run python scripts_edit/preprocess_edit.py "$bag_path" \
        --fps "$fps" \
        --width "$width" \
        --height "$height" \
        --depth-source "$depth_source" \
        --output-dir "$output_dir" > "logs/${log_file_name}.log" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: ${output_dir}"
    else
        echo "✗ Failed: ${output_dir} (exit code: $exit_code). Check logs/${log_file_name}.log for details."
    fi
    
    return $exit_code
}

export -f run_experiment

# GNU parallel을 사용할 수 있는지 확인
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution... (${PARALLEL_JOBS} jobs at a time)"
    # --bar: 진행률 표시
    # --joblog: 작업 로그 저장
    # --halt soon,fail=1: 하나의 작업이라도 실패하면 새 작업을 시작하지 않음
    printf '%s\n' "${all_experiments[@]}" | parallel -j "${PARALLEL_JOBS}" --bar --joblog logs/parallel_jobs.log --halt soon,fail=1 run_experiment
else
    echo "GNU Parallel not found. Running experiments sequentially..."
    # GNU Parallel이 없으면 순차 실행으로 대체
    for exp in "${all_experiments[@]}"; do
        run_experiment "$exp"
        if [ $? -ne 0 ]; then
            echo "Stopping script due to a failed experiment."
            exit 1
        fi
    done
fi

echo "All experiments completed!"
