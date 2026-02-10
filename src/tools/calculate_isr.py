import os
import glob
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

def calculate_isr_for_file(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'ID' not in df.columns or 'y_pred' not in df.columns:
            return None, "Missing 'ID' or 'y_pred' columns"

        # Group by ID
        grouped = df.groupby('ID')
        
        y1_list = []
        y2_list = []
        
        valid_subjects_count = 0
        
        for subject_id, group in grouped:
            if len(group) >= 2:
                # Randomly select 2 samples without replacement
                samples = group.sample(n=2, replace=False, random_state=42) 
                
                preds = samples['y_pred'].values
                y1_list.append(preds[0])
                y2_list.append(preds[1])
                valid_subjects_count += 1
                
        if valid_subjects_count < 2: # Need at least a few to calc kappa
            return None, f"Not enough subjects with >= 2 samples (Found {valid_subjects_count})"
            
        kappa = cohen_kappa_score(y1_list, y2_list, weights='quadratic')
        return kappa, None
        
    except Exception as e:
        return None, str(e)

def main():
    parser = argparse.ArgumentParser(description="Calculate ISR (Cohen's Kappa) for OOF predictions.")
    parser.add_argument("--result_dir", type=str, default="./result", help="Directory containing result folders")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save results as JSON")
    
    args = parser.parse_args()
    
    # search for all *_oof_predictions.csv
    search_pattern = os.path.join(args.result_dir, "**", "*_oof_predictions.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} prediction files in {args.result_dir}")
    
    results = []
    
    for file_path in tqdm(files, desc="Processing files"):
        kappa, error = calculate_isr_for_file(file_path)
        
        # Get relative path for readability
        rel_path = os.path.relpath(file_path, args.result_dir)
        
        if kappa is not None:
            results.append({
                "File": rel_path,
                "Kappa (Quadratic)": float(kappa), # Convert to float for JSON serializability
                "Status": "Success"
            })
        else:
             results.append({
                "File": rel_path,
                "Kappa (Quadratic)": None,
                "Status": f"Error: {error}"
            })

    # Sort results
    # Put errors at the end
    results.sort(key=lambda x: (x["Status"] != "Success", x["File"]))

    # Save to JSON if requested
    if args.output_json:
        try:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {args.output_json}")
        except Exception as e:
            print(f"\nError saving to JSON: {e}")

    # Print results (using pandas for nice table)
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*80)
        print("INTER-SAMPLE RELIABILITY (ISR) REPORT")
        print("="*80)
        # Adjust display options
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)
        
        print(df_res)
        print("="*80)
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
