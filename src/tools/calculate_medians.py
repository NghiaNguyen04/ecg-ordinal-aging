import pandas as pd
import numpy as np
import os

def calculate_medians():
    # Define paths
    subject_info_path = '/workspace/ecg-ordinal-aging/data/raw/autonomic-aging-a-dataset/subject-info.csv'
    reduced_groups_path = '/workspace/ecg-ordinal-aging/data/processed/Age_group_reduced.csv'

    if not os.path.exists(subject_info_path):
        print(f"File not found: {subject_info_path}")
        return
    if not os.path.exists(reduced_groups_path):
        print(f"File not found: {reduced_groups_path}")
        return

    # Load files
    subject_info = pd.read_csv(subject_info_path, dtype={'ID': str})
    reduced_groups = pd.read_csv(reduced_groups_path, dtype={'ID': str})

    # Merge
    merged = pd.merge(reduced_groups, subject_info, on='ID', how='inner')
    
    print(f"Merged {len(merged)} records.")

    # Mapping from Age_group to approximate median age based on ranges
    # 1 (18-19), 2 (20-24), 3 (25-29), 4 (30-34), 5 (35-39), 
    # 6 (40-44), 7 (45-49), 8 (50-54), 9 (55-59), 10 (60-64), 
    # 11 (65-69), 12 (70-74), 13 (75-79), 14 (80-84), 15 (85-92)
    age_mapping = {
        1: 18.5, # 18-19
        2: 22.0,   # 20-24
        3: 27.0,   # 25-29
        4: 32.0,   # 30-34
        5: 37.0,   # 35-39
        6: 42.0,   # 40-44
        7: 47.0,   # 45-49
        8: 52.0,   # 50-54
        9: 57.0,   # 55-59
        10: 62.0,  # 60-64
        11: 67.0,  # 65-69
        12: 72.0,  # 70-74
        13: 77.0,  # 75-79
        14: 82.0,  # 80-84
        15: 88.5   # 85-92
    }

    merged['estimated_age'] = merged['Age_group'].map(age_mapping)
    
    # Check for unmapped values
    if merged['estimated_age'].isnull().any():
        print("Warning: Some Age_groups could not be mapped:")
        print(merged[merged['estimated_age'].isnull()]['Age_group'].unique())

    # Calculate medians for each reduced group
    medians = merged.groupby('Age_group_reduced')['estimated_age'].median().sort_index()
    
    print("\nCalculated Medians for each reduced group:")
    print(medians)
    
    # Calculate means as well for reference
    means = merged.groupby('Age_group_reduced')['estimated_age'].mean().sort_index()
    print("\nCalculated Means for each reduced group:")
    print(means)
    
    print("\nSample counts per reduced group:")
    print(merged.groupby('Age_group_reduced')['ID'].count())

if __name__ == "__main__":
    calculate_medians()
