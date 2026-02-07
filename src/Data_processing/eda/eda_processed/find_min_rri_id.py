
import csv
import sys

file_path = r'ecg-ordinal-aging\data\processed\seg_300s\data_300s_order5.csv'

min_rri = float('inf')
min_id = None

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rri_indices = [i for i, col in enumerate(header) if col.startswith('RRI_')]
    
    for row in reader:
        for i in rri_indices:
            if i < len(row) and row[i]:
                try:
                    val = float(row[i])
                    if val < min_rri:
                        min_rri = val
                        min_id = row[0]
                except: pass

print(f"ID with Minimum RRI ({min_rri}): {min_id}")
