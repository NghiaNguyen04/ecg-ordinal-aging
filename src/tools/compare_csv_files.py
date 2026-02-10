
import pandas as pd
import os
import numpy as np

def compare_files(file1_path, file2_path):
    print(f"Comparing files:\n1. {file1_path}\n2. {file2_path}\n")

    if not os.path.exists(file1_path):
        print(f"Error: File not found: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Error: File not found: {file2_path}")
        return

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Compare Shapes
    print(f"Shape of File 1: {df1.shape}")
    print(f"Shape of File 2: {df2.shape}")

    if df1.shape != df2.shape:
        print("❌ Shapes are different.")
    else:
        print("✅ Shapes match.")

    # Compare Columns
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    
    if cols1 != cols2:
        print("❌ Column names or order are different.")
        diff_cols = set(cols1) ^ set(cols2)
        if diff_cols:
            print(f"   Differing columns: {diff_cols}")
    else:
        print("✅ Column names and order match.")

    # Compare Values
    # Align columns just in case order is different but content is same (if desired, but strict comparison usually expects same order)
    # For this check, we'll assume strict equality is desired.
    
    if df1.shape == df2.shape and cols1 == cols2:
        # Check for equality with some tolerance for float
        try:
            # Sort by ID and Segment_Order if available to ensure row alignment
            sort_cols = [c for c in ['ID', 'Segment_Order'] if c in df1.columns]
            if sort_cols:
                df1_sorted = df1.sort_values(by=sort_cols).reset_index(drop=True)
                df2_sorted = df2.sort_values(by=sort_cols).reset_index(drop=True)
            else:
                df1_sorted = df1
                df2_sorted = df2

            # Compare
            # Using basic equality
            equals = df1_sorted.equals(df2_sorted)
            if equals:
                print("✅ Content is exactly identical.")
            else:
                print("❌ Content is NOT identical.")
                
                # Check for numerical closeness
                try:
                    pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False, rtol=1e-5, atol=1e-5)
                    print("⚠️ Content is numerically similar (within tolerance 1e-5). Differences might be due to float precision.")
                except AssertionError as e:
                    print("❌ Content is significantly different.")
                    print(e)
                    
        except Exception as e:
            print(f"Error during value comparison: {e}")

    else:
        print("Skipping detailed value comparison because structure differs.")

if __name__ == "__main__":
    # Define paths
    # Using absolute paths or relative to project root
    base_dir = r"D:\OneDrive\Desktop\The big project\ecg-ordinal-aging"
    
    file_new = os.path.join(base_dir, "data", "processed", "seg_300s", "data_300s_order5_new.csv")
    file_old = os.path.join(base_dir, "data", "processed", "seg_300s", "data_300s_order5.csv")
    
    compare_files(file_new, file_old)
