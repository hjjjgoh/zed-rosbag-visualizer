"""
로깅 설정 유틸리티

실험 추적을 위한 로깅 설정 함수를 제공
로그는 콘솔과 파일에 동시 출력되며, 파일명은 output 디렉토리 이름과 동기화
예: output/my_exp → logs/my_exp_20241030.log
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional
import re

def setup_logger(
    output_dir: Path,
    log_level: int = logging.INFO,
    log_dir: str = "logs"
) -> tuple[logging.Logger, Path]:
    """
    로거 설정 및 초기화
    
    Args:
        output_dir: 출력 디렉토리 경로 (로그 파일명 생성에 사용)
        log_level: 로그 레벨 (기본값: logging.INFO)
        log_dir: 로그 파일을 저장할 디렉토리 (기본값: "logs")
    
    Returns:
        tuple[logging.Logger, Path]: 설정된 로거와 로그 파일 경로
    
    Example:
        >>> logger, log_file = setup_logger(Path("output/my_exp"))
        >>> logger.info("Processing started")
        # 콘솔: 2024-10-30 14:30:20,123 - INFO - Processing started
        # 파일: logs/my_exp_20241030.log에 동일 내용 기록
    """
    
    # output 디렉토리 이름 추출 (예: "output/my_exp" → "my_exp")
    out_dir_name = output_dir.name
    
    # 디렉토리 이름에 6자리 숫자(YYMMDD 형식)가 있는지 확인하여 로그 파일명 결정
    if re.search(r'\d{6}', out_dir_name):
        log_filename = f"{out_dir_name}.log"
    else:
        # 날짜 타임스탬프 생성 (YYMMDD 형식)
        timestamp = time.strftime("%y%m%d")
        log_filename = f"{out_dir_name}_{timestamp}.log"
    
    log_file = Path(log_dir) / log_filename
    
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 기존 핸들러 제거 (중복 방지)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 기본 로거 설정
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),  # 파일 핸들러
            logging.StreamHandler(sys.stdout)  # 콘솔 핸들러
        ],
        force=True  # 기존 설정 덮어쓰기
    )
    
    logger = logging.getLogger()
    
    # 시작 메시지
    logger.info("="*60)
    logger.info("PREPROCESSING STARTED")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)
    
    return logger, log_file


def log_summary(
    logger: logging.Logger,
    log_file: Path,
    out_dir: Path,
    stats: dict,
    total_time: float
):
    """
    처리 완료 후 요약 로그 출력
    
    Args:
        logger: 로거 인스턴스
        log_file: 로그 파일 경로
        out_dir: 출력 디렉토리
        stats: 통계 정보 딕셔너리
        total_time: 총 처리 시간 (초)
    
    Example:
        >>> stats = {
        ...     'total_left': 327,
        ...     'total_right': 327,
        ...     'processed_pairs': 327,
        ...     'match_rate': 100.0,
        ...     'fps': 15.1,
        ...     'tolerance_ns': 200_000_000,
        ...     'collection_time': 7.15,
        ...     'processing_time': 145.23
        ... }
        >>> log_summary(logger, log_file, out_dir, stats, 152.38)
    """
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE - SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Resolution: {stats.get('width', 0)}x{stats.get('height', 0)} @ {stats.get('fps', 0)} FPS")
    logger.info(f"Intrinsics: fx={stats.get('fx', 0):.2f}, fy={stats.get('fy', 0):.2f}, cx={stats.get('cx', 0):.2f}, cy={stats.get('cy', 0):.2f}")
    logger.info(f"Stereo baseline: {stats.get('baseline_m', 0)} m")
    logger.info("-" * 20) # 구분선
    
    logger.info(f"Input frames - left: {stats.get('total_left', 0)}, right: {stats.get('total_right', 0)}")
    
    if 'synchronized_points' in stats:
        logger.info(f"Synchronized trajectory points: {stats['synchronized_points']}")

    if 'processed_pairs' in stats:
        logger.info(f"Matched stereo pairs: {stats['processed_pairs']} ({stats.get('match_rate', 0):.1f}%)")
        logger.info(f"Written frames: {stats.get('written_frames', 0)} frames saved as RGB/Depth video")
    
    logger.info(f"Output video FPS: {stats.get('fps', 0)}")
    logger.info(f"Stereo matching tolerance: {stats.get('tolerance_ns', 0)} ns")
    logger.info(f"Collection time: {stats.get('collection_time', 0):.2f}s")
    logger.info(f"Processing time: {stats.get('processing_time', 0):.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Log file saved: {log_file}")
    logger.info(f"Trajectory file saved: {out_dir / 'trajectory.json'}")
    logger.info("="*60)

