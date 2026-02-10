import os
import glob
import json
import argparse
from datetime import datetime

def aggregate_results(result_dir, output_file):
    search_pattern = os.path.join(result_dir, "**", "*_cv_stats.json")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} CV stats files in {result_dir}")
    
    aggregated_data = {}
    
    for file_path in files:
        try:
            # Extract experiment name from directory structure
            # Example path: result/Coral_bmi/Resnet34_coral_20260207-122839/Resnet34_coral_20260207-122839_cv_stats.json
            parts = os.path.normpath(file_path).split(os.sep)
            
            # Find the 'result' directory index
            try:
                result_idx = parts.index('result')
                experiment_name = parts[result_idx + 1]
                run_folder = parts[result_idx + 2]
            except (ValueError, IndexError):
                # Fallback if structure is different
                experiment_name = parts[-3]
                run_folder = parts[-2]
            
            with open(file_path, 'r') as f:
                stats = json.load(f)
            
            if experiment_name not in aggregated_data:
                aggregated_data[experiment_name] = {}
            
            # Flatten the stats for easier viewing/JSON
            flattened_stats = {}
            for metric, values in stats.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        flattened_stats[f"{metric}_{key}"] = val
                else:
                    flattened_stats[metric] = values
            
            aggregated_data[experiment_name][run_folder] = flattened_stats
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(aggregated_data, f, indent=4)
    
    print(f"Aggregated results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CV results from experiment folders.")
    parser.add_argument("--result_dir", type=str, default="./result", help="Root directory for experiments")
    parser.add_argument("--output_file", type=str, default="./logs/aggregated_cv_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    aggregate_results(args.result_dir, args.output_file)
