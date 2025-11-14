"""
비디오 포맷 변환 유틸리티

FFmpeg를 사용하여 mp4v 코덱 비디오를 H.264 코덱으로 변환합니다.
변환된 파일은 원본 파일을 대체합니다.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def convert_videos_to_h264(
    output_dir: Path,
    video_files: Optional[List[str]] = None,
    crf: int = 20,
    preset: str = "medium",
    verbose: bool = True
) -> None:
    """
    mp4v로 작성된 비디오 파일들을 FFmpeg로 H.264로 변환하고 원본 교체
    
    Args:
        output_dir: 비디오 파일들이 있는 디렉토리
        video_files: 변환할 파일 리스트 (None이면 기본 파일들)
        crf: 품질 설정 (0-51, 낮을수록 고품질, 기본값: 20)
        preset: 인코딩 속도 (ultrafast, fast, medium, slow, veryslow, 기본값: medium)
        verbose: 진행 상황 출력 여부
    
    Note:
        - FFmpeg가 시스템 PATH에 설치되어 있어야 합니다
        - 변환 실패 시 원본 파일은 유지됩니다
        - 임시 파일을 사용하여 안전하게 변환합니다
    """
    # FFmpeg 설치 확인
    if shutil.which("ffmpeg") is None:
        if verbose:
            print("\n[WARN] ffmpeg not found in PATH — skipping conversion.")
            print("Install FFmpeg or run manually:")
            print("  ffmpeg -y -i input.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 output_h264.mp4")
        return
    
    # 변환 대상 파일 리스트
    if video_files is None:
        video_files = [
            "rgb.mp4",
            "depth_foundation.mp4",
            "depth_foundation_color.mp4",
            "depth_zed.mp4",
        ]
    
    if verbose:
        print("\n[FFMPEG] Converting to H.264...")
    
    for src_name in video_files:
        src = output_dir / src_name
        if not src.exists():
            continue
        
        # 임시 파일로 변환
        tmp = output_dir / f"{src_name}.tmp.mp4"
        
        # FFmpeg 명령어 구성
        cmd = [
            "ffmpeg",
            "-y",  # 덮어쓰기 확인 없이
            "-i", str(src),  # 입력 파일
            "-c:v", "libx264",  # H.264 코덱
            "-pix_fmt", "yuv420p",  # 픽셀 포맷 (호환성)
            "-crf", str(crf),  # 품질 (20 = 고품질)
            "-preset", preset,  # 인코딩 속도
            "-movflags", "+faststart",  # 웹 스트리밍 최적화
            str(tmp),  # 출력 파일
        ]
        
        if verbose:
            print(f"[converting] {src_name} → H.264...")
        
        # FFmpeg 실행
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 변환 결과 처리
        if result.returncode == 0 and tmp.exists():
            # 변환 성공 → 원본 삭제 후 임시 파일을 원본 이름으로 변경
            src.unlink()
            tmp.rename(src)
            if verbose:
                print(f"[ok] {src_name} converted to H.264 (original mp4v replaced)")
        else:
            # 변환 실패 → 임시 파일 삭제 (있다면)
            if tmp.exists():
                tmp.unlink()
            if verbose:
                print(f"[fail] conversion failed for {src_name} (original kept)")


def check_ffmpeg_installed() -> bool:
    """
    FFmpeg가 시스템에 설치되어 있는지 확인
    
    Returns:
        bool: 설치 여부
    """
    return shutil.which("ffmpeg") is not None


def get_video_info(video_path: Path) -> Optional[dict]:
    """
    비디오 파일의 정보를 가져옵니다 (ffprobe 사용)
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        dict: 비디오 정보 (코덱, 해상도, FPS 등) 또는 None
    """
    if not shutil.which("ffprobe"):
        return None
    
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            str(video_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 간단한 파싱 (실제로는 더 정교한 파싱이 필요)
        lines = result.stdout.strip().split('\n')
        info = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value
        
        return info
    except Exception:
        return None

