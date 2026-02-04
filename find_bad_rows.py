
import csv
import os

file_path = r'd:\OneDrive\Desktop\The big project\ecg-ordinal-aging\data\processed\seg_300s\data_300s_order5.csv'

print(f"Checking for bad RRI rows in {os.path.basename(file_path)}...")

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    rri_indices = [i for i, col in enumerate(header) if col.startswith('RRI_')]
    
    bad_rows = []
    
    for row_idx, row in enumerate(reader, start=2):
        has_bad = False
        min_v = float('inf')
        max_v = float('-inf')
        
        for i in rri_indices:
            if i < len(row) and row[i]:
                try:
                    v = float(row[i])
                    if v <= 0 or v > 3.0: # Threshold for bad RRI
                        has_bad = True
                    if v < min_v: min_v = v
                    if v > max_v: max_v = v
                except: pass
        
        if has_bad:
            bad_rows.append({
                'Row': row_idx,
                'ID': row[0],
                'MinRRI': min_v,
                'MaxRRI': max_v
            })
            if len(bad_rows) >= 10: break

print("-" * 30)
if bad_rows:
    print(f"Found {len(bad_rows)}+ rows with RRI anomalies (<=0 or >3.0):")
    for b in bad_rows:
        print(f"Row {b['Row']}, ID: {b['ID']}, Range: [{b['MinRRI']:.4f}, {b['MaxRRI']:.4f}]")
else:
    print("No bad rows found with threshold.")
