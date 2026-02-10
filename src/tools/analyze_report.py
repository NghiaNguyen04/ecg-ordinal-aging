
import csv
import math
import sys
import os

file_path = r'd:\OneDrive\Desktop\The big project\ecg-ordinal-aging\data\processed\seg_300s\data_300s_order5.csv'
report_path = r'd:\OneDrive\Desktop\The big project\ecg-ordinal-aging\analysis_report.md'

with open(report_path, 'w', encoding='utf-8') as report:
    report.write(f"# Analysis of {os.path.basename(file_path)}\n\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            idx_bmi = -1
            idx_sex = -1
            idx_age = -1
            
            if 'BMI' in header: idx_bmi = header.index('BMI')
            if 'Sex' in header: idx_sex = header.index('Sex')
            if 'Age_group' in header: idx_age = header.index('Age_group')

            rri_indices = [i for i, col in enumerate(header) if col.startswith('RRI_')]
            hrv_indices = [i for i, col in enumerate(header) if col.startswith('HRV_')]
            
            row_count = 0
            missing_counts = {} 
            inf_counts = {}
            nan_counts = {}
            
            sex_counts = {}
            age_counts = {}
            
            rri_min = float('inf')
            rri_max = float('-inf')
            rri_zero_count = 0
            
            bmi_min = float('inf')
            bmi_max = float('-inf')
            bmi_sum = 0
            bmi_count = 0
            bmi_zeros = 0
            
            hrv_nan_total = 0
            hrv_inf_total = 0

            for row in reader:
                row_count += 1
                
                # Missing check
                for i, val in enumerate(row):
                    if not val:
                        missing_counts[i] = missing_counts.get(i, 0) + 1
                        if i in hrv_indices: hrv_nan_total += 1
                
                # RRI Stats
                for i in rri_indices:
                    if i < len(row) and row[i]:
                        try:
                            v = float(row[i])
                            if v < rri_min: rri_min = v
                            if v > rri_max: rri_max = v
                            if v <= 0: rri_zero_count += 1
                        except: pass
                
                # BMI Stats
                if idx_bmi != -1 and idx_bmi < len(row) and row[idx_bmi]:
                    try:
                        v = float(row[idx_bmi])
                        if v < bmi_min: bmi_min = v
                        if v > bmi_max: bmi_max = v
                        bmi_sum += v
                        bmi_count += 1
                        if v <= 0: bmi_zeros += 1
                    except: pass

                # Sex Stats
                if idx_sex != -1 and idx_sex < len(row):
                    v = row[idx_sex]
                    sex_counts[v] = sex_counts.get(v, 0) + 1

                # Age Stats
                if idx_age != -1 and idx_age < len(row):
                    v = row[idx_age]
                    age_counts[v] = age_counts.get(v, 0) + 1
                    
                # HRV Inf/NaN check
                for i in hrv_indices:
                    if i < len(row) and row[i]:
                        try:
                            v = float(row[i])
                            if math.isinf(v):
                                inf_counts[i] = inf_counts.get(i, 0) + 1
                                hrv_inf_total += 1
                            elif math.isnan(v):
                                nan_counts[i] = nan_counts.get(i, 0) + 1
                                hrv_nan_total += 1
                        except: pass

        report.write(f"Total Rows: {row_count}\n\n")
        
        report.write("## Missing Values\n")
        if missing_counts:
            for i, count in missing_counts.items():
                report.write(f"- {header[i]}: {count}\n")
        else:
            report.write("None\n")
        report.write("\n")

        report.write("## Infinite Values\n")
        if inf_counts:
            for i, count in inf_counts.items():
                report.write(f"- {header[i]}: {count}\n")
        else:
            report.write("None\n")
        report.write("\n")
        
        report.write("## Distributions\n")
        report.write(f"- Sex Counts: {sex_counts}\n")
        report.write(f"- Age Group Counts: {dict(sorted(age_counts.items()))}\n")
        report.write("\n")
        
        report.write("## BMI Statistics\n")
        if bmi_count > 0:
            report.write(f"- Min: {bmi_min:.2f}\n")
            report.write(f"- Max: {bmi_max:.2f}\n")
            report.write(f"- Avg: {bmi_sum/bmi_count:.2f}\n")
            report.write(f"- BMI <= 0 Count: {bmi_zeros}\n")
        else:
            report.write("No BMI data.\n")
        report.write("\n")
        
        report.write("## RRI Statistics\n")
        report.write(f"- Min: {rri_min}\n")
        report.write(f"- Max: {rri_max}\n")
        report.write(f"- RRI <= 0 Count: {rri_zero_count}\n")
        report.write("\n")
        
        report.write("## HRV Anomalies\n")
        report.write(f"- Total HRV NaNs: {hrv_nan_total}\n")
        report.write(f"- Total HRV Infs: {hrv_inf_total}\n")

    except Exception as e:
        report.write(f"Error: {e}\n")
