# PowerShell을 사용하여 rosbag2_4 데이터에 대한 전처리 실험을 순차적으로 실행하는 스크립트

# --- 설정 ---
Write-Output "실험 설정을 로드합니다..."

# --- 실험 정의 ---
# 형식: 해시 테이블(딕셔너리)의 배열로 각 실험을 정의합니다.
# NOTE: Foundation Stereo 모델의 제약으로 인해 모든 해상도는 16의 배수여야 함
$experiments = @(
    # 실험 그룹 1: FPS 변경 (해상도 1088x720, depth=both 고정)
    @{ bag_path="data/rosbag2_5"; fps=5;  width=1280; height=720; depth_source="both"; output_dir="output/data5_fps/exp_fps_5_1280x720" },
    @{ bag_path="data/rosbag2_5"; fps=10; width=1280; height=720; depth_source="both"; output_dir="output/data5_fps/exp_fps_10_1280x720" },
    @{ bag_path="data/rosbag2_5"; fps=15; width=1280; height=720; depth_source="both"; output_dir="output/data5_fps/exp_fps_15_1280x720" },

    # 실험 그룹 2: 해상도 변경 (FPS=15, depth=both 고정, 16:9 비율)
    # 너비를 2의 거듭제곱으로, 높이는 16:9 비율을 유지하며 16의 배수로 맞춤
    @{ bag_path="data/rosbag2_5"; fps=15; width=512;  height=288;  depth_source="both"; output_dir="output/data5_resolution/exp_res_512x288" },
    @{ bag_path="data/rosbag2_5"; fps=15; width=1024; height=576;  depth_source="both"; output_dir="output/data5_resolution/exp_res_1024x576" },
    @{ bag_path="data/rosbag2_5"; fps=15; width=2048; height=1152; depth_source="both"; output_dir="output/data5_resolution/exp_res_2048x1152" }
)

# --- 실행 로직 ---

# 로그 및 출력 디렉토리 생성
$log_dir = "logs"
if (-not (Test-Path $log_dir)) { New-Item -ItemType Directory -Path $log_dir }
if (-not (Test-Path "output/data4_fps")) { New-Item -ItemType Directory -Path "output/data4_fps" }
if (-not (Test-Path "output/data4_resolution")) { New-Item -ItemType Directory -Path "output/data4_resolution" }

# 각 실험을 순차적으로 실행
foreach ($exp in $experiments) {
    # PowerShell 스크립트에 의해 생성되던 로그 파일 관련 로직 제거
    # $log_file_name = Split-Path -Leaf $exp.output_dir
    # $log_path = Join-Path $log_dir "$($log_file_name).log"
    
    Write-Output "---"
    Write-Output "Starting: Bag=$($exp.bag_path), FPS=$($exp.fps), Res=$($exp.width)x$($exp.height), Depth=$($exp.depth_source)"
    
    # 명령어 실행. Tee-Object를 제거하여 PowerShell 로그 파일 생성을 중단
    # 출력은 콘솔에만 표시됩니다. 파이썬 내부 로거는 계속 파일에 기록
    & uv run python scripts_edit/preprocess_edit.py $exp.bag_path --fps $exp.fps --width $exp.width --height $exp.height --depth-source $exp.depth_source --output-dir $exp.output_dir
    
    # $LASTEXITCODE는 마지막으로 실행된 프로그램의 종료 코드를 담고 있음
    if ($LASTEXITCODE -eq 0) {
        Write-Host -ForegroundColor Green "✓ Completed: $($exp.output_dir)"
    } else {
        # 오류 메시지에서 PowerShell이 생성하던 로그 파일 경로 제거
        # 파이썬 스크립트가 시작될 때 로그 경로 출력하므로 콘솔에서 확인 가능
        Write-Host -ForegroundColor Red "✗ Failed: $($exp.output_dir) (exit code: $LASTEXITCODE). Check console output for log file path and details."
        Write-Host -ForegroundColor Red "스크립트를 중단합니다."
        # 스크립트 실행 중단
        exit $LASTEXITCODE
    }
}

Write-Host -ForegroundColor Green "---"
Write-Host -ForegroundColor Green "모든 실험이 성공적으로 완료되었습니다!"
