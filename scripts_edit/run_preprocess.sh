#!bin/bash

for fps in 5 10 15; do
    uv run python scripts_edit/preprocess_edit.py .\data\rosbag2_4 --fps $fps --depth-source foundation --matching-algorithm batch --output-dir .\output\rosbag2_4_optimized >> logs/log_$fps.txt
done

