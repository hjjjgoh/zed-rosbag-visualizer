"""
연속된 이미지 파일(PNG)을 MP4 비디오로 변환하는 스크립트

[전제 조건]
- 이미지 파일들은 파일 이름순으로 정렬했을 때 올바른 순서가 되어야 합니다.
  (예: frame_001.png, frame_002.png, frame_003.png ...)

[필요 라이브러리]
- opencv-python

[기본 사용법]
  uv run python tools/images_to_video.py <이미지_폴더_경로> <출력_비디오_경로> --fps <프레임_속도>

[예제]
  # 'my_frames' 폴더의 png들을 'output.mp4'로, 초당 30프레임으로 만들기
  uv run python tools/images_to_video.py my_frames output.mp4 --fps 30
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def create_video_from_images(image_dir: Path, output_video_path: Path, fps: float):
    """
    지정된 디렉토리의 PNG 이미지들로부터 비디오를 생성합니다.
    """
    # 1. 이미지 디렉토리 존재 여부 확인
    if not image_dir.is_dir():
        print(f"오류: 이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return

    # 2. 이미지 파일 목록 가져오기 및 정렬
    # 파일 이름순으로 정렬하여 프레임 순서를 보장합니다.
    image_files = sorted(list(image_dir.glob("*.png")))
    
    if not image_files:
        print(f"오류: '{image_dir}' 디렉토리에서 PNG 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    # 3. 첫 번째 이미지를 읽어 비디오의 너비와 높이 결정
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"오류: 첫 번째 이미지 파일을 읽을 수 없습니다: {image_files[0]}")
        return
    height, width, _ = first_image.shape
    
    # 4. 비디오 라이터(VideoWriter) 객체 설정
    #   - 'avc1'은 H.264 코덱을 의미하며, MP4 파일에 널리 사용됩니다.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # H.264 코덱(avc1)을 사용할 수 없는 경우 'mp4v'로 대체(fallback) 시도
    if not video_writer.isOpened():
        print("경고: H.264(avc1) 코덱을 사용할 수 없습니다. 'mp4v' 코덱으로 다시 시도합니다.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"오류: 비디오 파일({output_video_path})을 열 수 없습니다. 시스템에 사용 가능한 코덱이 없는 것 같습니다.")
        return

    # 5. 모든 이미지를 순서대로 읽어 비디오 프레임으로 추가
    print(f"비디오를 생성하는 중... -> {output_video_path}")
    for image_file in tqdm(image_files, desc="Processing images"):
        frame = cv2.imread(str(image_file))
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"경고: 이미지 파일을 읽는 데 실패했습니다: {image_file}")

    # 6. 작업 완료 후 리소스 해제
    video_writer.release()
    print("\n비디오 생성이 성공적으로 완료되었습니다!")

def main():
    parser = argparse.ArgumentParser(description="Convert a sequence of PNG images to an MP4 video.")
    parser.add_argument("image_dir", type=str, help="이미지 파일들이 있는 디렉토리 경로")
    parser.add_argument("output_video_path", type=str, help="생성될 MP4 비디오 파일의 경로")
    parser.add_argument("--fps", type=float, default=30.0, help="비디오의 초당 프레임 수 (기본값: 30)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_video_path = Path(args.output_video_path)

    # 출력 비디오의 부모 디렉토리가 없으면 생성
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    create_video_from_images(image_dir, output_video_path, args.fps)

if __name__ == "__main__":
    main()
