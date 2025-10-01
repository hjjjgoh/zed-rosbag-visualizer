import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

dinov2_path = project_root / "src" / "foundation_stereo" / "dinov2"
sys.path.insert(0, str(dinov2_path))


import argparse
import yaml
from src.player import run

def main():
    parser = argparse.ArgumentParser(description="Rerun ZED(ROS2) visualizer.")
    parser.add_argument("bag_path", type=str, help="Path to the ROS bag directory.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    bag_path = Path(args.bag_path)
    config_path = Path(args.config)

    # 설정 파일을 직접 읽어옵니다.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run(bag_path, config)

if __name__ == "__main__":
    main()