import argparse
from pathlib import Path
from rosbags.rosbag2 import Reader
import json
import csv

def get_rosbag_topics(bag_path):
    topics = {}
    with Reader(bag_path) as reader:
        for conn in reader.connections:
            topics[conn.topic] = {
                'type': conn.msgtype,
                'count': conn.msgcount
            }
    return topics

def main():
    parser = argparse.ArgumentParser(description='List topics in ROS bag files and save to JSON/CSV.')
    parser.add_argument('bag_dirs', nargs='+', help='List of ROS bag directories.')
    parser.add_argument('--output-json', default='data/analyze/rosbag_topics.json', help='Output JSON file name.')
    parser.add_argument('--output-csv', default='data/analyze/rosbag_topics.csv', help='Output CSV file name.')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    all_bags_info = {}
    print("Processing rosbag directories...")
    for bag_dir in args.bag_dirs:
        p_bag_dir = Path(bag_dir)
        if p_bag_dir.is_dir():
            print(f"Reading topics from: {p_bag_dir.name}")
            try:
                all_bags_info[p_bag_dir.name] = get_rosbag_topics(p_bag_dir)
            except Exception as e:
                print(f"  Error processing {p_bag_dir.name}: {e}")
    
    # Save to JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_bags_info, f, indent=2, ensure_ascii=False)
    print(f"\nTopic information saved to {args.output_json}")

    # Save to CSV
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Bag Name', 'Topic', 'Message Count', 'Message Type'])
            for bag_name, topics in all_bags_info.items():
                if not topics:
                    writer.writerow([bag_name, 'N/A', 0, 'N/A'])
                for topic, info in sorted(topics.items()):
                    writer.writerow([bag_name, topic, info['count'], info['type']])
        print(f"Topic information also saved to {args.output_csv}")
    except Exception as e:
        print(f"  Error saving to CSV: {e}")

    # Print summary to console
    for bag_name, topics in all_bags_info.items():
        print(f'\n--- Topics in {bag_name} ---')
        if not topics:
            print("  No topics found.")
            continue
        for topic, info in sorted(topics.items()):
            print(f"  - {topic}")
            print(f"    - Count: {info['count']} msgs")
            print(f"    - Type:  {info['type']}")
        

if __name__ == '__main__':
    main()
