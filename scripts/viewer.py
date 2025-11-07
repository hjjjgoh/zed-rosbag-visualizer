"""
Main entry point for ZED ROS bag preprocessing and visualization
"""
import autorootcwd  # 프로젝트 루트로 자동 이동
import sys
import argparse
import yaml
from pathlib import Path

# dinov2 서브모듈을 sys.path에 추가 (dinov2 내부에서 절대 import 사용)
project_root = Path.cwd()
dinov2_path = project_root / "src" / "foundation_stereo" / "dinov2"
if str(dinov2_path) not in sys.path:
    sys.path.insert(0, str(dinov2_path))

from scripts.preprocess import ZEDPreprocessor



def main():
    """Main entry point for preprocessing script"""
    parser = argparse.ArgumentParser(
        description="Preprocess ZED ROS bag data for tomato detection/segmentation pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "bag_path",
        type=str,
        help="Path to the ROS bag directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--depth-source",
        type=str,
        choices=['zed', 'foundation', 'both'],
        default='foundation',
        help="Depth data source: 'zed' (from ZED camera), 'foundation' (from FoundationStereo), or 'both' (default: foundation)"
    )
    
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable background segmentation"
    )
    
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Enable point cloud visualization"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for preprocessed data (optional)"
    )
    
    args = parser.parse_args()
    
    # Parse paths
    bag_path = Path(args.bag_path)
    config_path = Path(args.config)
    
    if not bag_path.exists():
        print(f"Error: ROS bag path does not exist: {bag_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.no_segmentation:
        config["visualization"]["use_segmentation"] = False
    
    # Create preprocessor and run
    preprocessor = ZEDPreprocessor(
        bag_path=bag_path,
        config=config,
        depth_source=args.depth_source,
        use_pointcloud=args.pointcloud
    )
    
    preprocessor.run(
        output_dir=Path(args.output_dir) if args.output_dir else None
    )


if __name__ == "__main__":
    main()
