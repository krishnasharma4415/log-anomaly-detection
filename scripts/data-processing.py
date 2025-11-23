import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
DATASET_PATH = PROJECT_ROOT / "dataset"
LABELED_DATA_PATH = DATASET_PATH / "labeled_data"

LABEL_MAP = {
    0: 'normal',
    1: 'anomaly'  # All anomaly types combined
}

# Original 7-class mapping for reference
ORIGINAL_LABEL_MAP = {
    0: 'normal',
    1: 'security_anomaly',
    2: 'system_failure',
    3: 'performance_issue',
    4: 'network_anomaly',
    5: 'config_error',
    6: 'hardware_issue'
}

print("Class Labels:")
for label_id, label_name in LABEL_MAP.items():
    print(f"  {label_id}: {label_name}")
print(f"Dataset path: {LABELED_DATA_PATH}")
# Timestamp parsing functions
def parse_android_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        current_year = datetime.now().year
        dt = datetime.strptime(f"{current_year}-{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return None

def parse_apache_timestamp(row):
    try:
        time_str = str(row['Time']).strip()
        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_bgl_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        dt = datetime.strptime(date_str, "%Y.%m.%d")
        return dt.strftime("%Y-%m-%d 00:00:00.000")
    except:
        return None

def parse_hadoop_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip().replace(',', '.')
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return None

def parse_hdfs_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        year = "20" + date_str[:2]
        month = date_str[2:4]
        day = date_str[4:6]
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_healthapp_timestamp(row):
    try:
        time_str = str(row['Time']).strip()
        parts = time_str.split(':')
        if len(parts) >= 4:
            time_str = ':'.join(parts[:-1]) + '.' + parts[-1]
        dt = datetime.strptime(time_str, "%Y%m%d-%H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return None

def parse_hpc_timestamp(row):
    try:
        timestamp = int(str(row['Time']).strip())
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_linux_timestamp(row):
    try:
        month_str = str(row['Month']).strip()
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        current_year = datetime.now().year
        dt = datetime.strptime(f"{current_year} {month_str} {date_str} {time_str}", "%Y %b %d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_mac_timestamp(row):
    try:
        month_str = str(row['Month']).strip()
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        current_year = datetime.now().year
        dt = datetime.strptime(f"{current_year} {month_str} {date_str} {time_str}", "%Y %b %d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_openssh_timestamp(row):
    try:
        month_str = str(row['Date']).strip()
        day_str = str(row['Day']).strip()
        time_str = str(row['Time']).strip()
        current_year = datetime.now().year
        dt = datetime.strptime(f"{current_year} {month_str} {day_str} {time_str}", "%Y %b %d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_openstack_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return None

def parse_proxifier_timestamp(row):
    try:
        time_str = str(row['Time']).strip()
        current_year = datetime.now().year
        dt = datetime.strptime(f"{current_year}.{time_str}", "%Y.%m.%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_spark_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        dt = datetime.strptime(f"20{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_thunderbird_timestamp(row):
    try:
        if 'Month' in row and 'Day' in row and 'Time' in row:
            month_str = str(row['Month']).strip()
            day_str = str(row['Day']).strip()
            time_str = str(row['Time']).strip()
            current_year = datetime.now().year
            dt = datetime.strptime(f"{current_year} {month_str} {day_str} {time_str}", "%Y %b %d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S.000")
        elif 'Date' in row:
            date_str = str(row['Date']).strip()
            dt = datetime.strptime(date_str, "%Y.%m.%d")
            return dt.strftime("%Y-%m-%d 00:00:00.000")
        return None
    except:
        return None

def parse_windows_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip()
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S.000")
    except:
        return None

def parse_zookeeper_timestamp(row):
    try:
        date_str = str(row['Date']).strip()
        time_str = str(row['Time']).strip().replace(',', '.')
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return None
def detect_log_type(filename):
    filename = filename.lower()
    if 'android' in filename:
        return 'android'
    elif 'apache' in filename:
        return 'apache'
    elif 'bgl' in filename:
        return 'bgl'
    elif 'hadoop' in filename:
        return 'hadoop'
    elif 'hdfs' in filename:
        return 'hdfs'
    elif 'health' in filename:
        return 'healthapp'
    elif 'hpc' in filename:
        return 'hpc'
    elif 'linux' in filename:
        return 'linux'
    elif 'mac' in filename:
        return 'mac'
    elif 'openssh' in filename:
        return 'openssh'
    elif 'openstack' in filename:
        return 'openstack'
    elif 'proxifier' in filename:
        return 'proxifier'
    elif 'spark' in filename:
        return 'spark'
    elif 'thunderbird' in filename:
        return 'thunderbird'
    elif 'windows' in filename:
        return 'windows'
    elif 'zookeeper' in filename or 'zookeper' in filename:
        return 'zookeeper'
    else:
        return 'unknown'

timestamp_parsers = {
    'android': parse_android_timestamp,
    'apache': parse_apache_timestamp,
    'bgl': parse_bgl_timestamp,
    'hadoop': parse_hadoop_timestamp,
    'hdfs': parse_hdfs_timestamp,
    'healthapp': parse_healthapp_timestamp,
    'hpc': parse_hpc_timestamp,
    'linux': parse_linux_timestamp,
    'mac': parse_mac_timestamp,
    'openssh': parse_openssh_timestamp,
    'openstack': parse_openstack_timestamp,
    'proxifier': parse_proxifier_timestamp,
    'spark': parse_spark_timestamp,
    'thunderbird': parse_thunderbird_timestamp,
    'windows': parse_windows_timestamp,
    'zookeeper': parse_zookeeper_timestamp
}
csv_files = list(LABELED_DATA_PATH.glob("*_labeled.csv"))
print(f"Found {len(csv_files)} labeled CSV files")
processed_files = {}
source_class_analysis = {}

for file_path in sorted(csv_files):
    print(f"\n{'='*80}")
    print(f"Processing: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    log_type = detect_log_type(file_path.name)
    print(f"Detected log type: {log_type}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataframe: {df.shape}")
        
        if log_type != 'unknown' and log_type in timestamp_parsers:
            parser_func = timestamp_parsers[log_type]
            df['timestamp_normalized'] = df.apply(parser_func, axis=1)
            
            successful = df['timestamp_normalized'].notna().sum()
            total = len(df)
            print(f"Normalized timestamps: {successful}/{total} ({successful/total*100:.1f}%)")
            
            df['timestamp_dt'] = pd.to_datetime(df['timestamp_normalized'], errors='coerce')
            
            df['hour'] = df['timestamp_dt'].dt.hour
            df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
            df['day_of_month'] = df['timestamp_dt'].dt.day
            df['month'] = df['timestamp_dt'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
            df['is_night'] = df['hour'].between(0, 6).astype(int)
            
            df = df.sort_values('timestamp_dt').reset_index(drop=True)
            df['time_diff_seconds'] = df['timestamp_dt'].diff().dt.total_seconds().fillna(0)
            
            df['log_index'] = range(len(df))
            df['logs_last_10'] = df.groupby(pd.Grouper(key='timestamp_dt', freq='1min'))['log_index'].transform('count')
            
            if 'AnomalyLabel' in df.columns:
                # Convert to binary: 0=normal, 1=anomaly (any non-zero becomes 1)
                df['AnomalyLabel'] = df['AnomalyLabel'].fillna(0).astype(int).clip(0, 6)
                df['AnomalyLabel'] = (df['AnomalyLabel'] > 0).astype(int)
                
                unique_labels = df['AnomalyLabel'].unique()
                present_classes = sorted([int(x) for x in unique_labels])
                missing_classes = [i for i in range(2) if i not in present_classes]  # Binary: 0,1
                
                label_counts = df['AnomalyLabel'].value_counts().sort_index()
                
                print(f"\n📊 BINARY CLASS DISTRIBUTION ANALYSIS:")
                print(f"  Classes present: {len(present_classes)}/2")
                print(f"  Present: {[LABEL_MAP[i] for i in present_classes]}")
                if missing_classes:
                    print(f"  ⚠️  Missing: {[LABEL_MAP[i] for i in missing_classes]}")
                
                print(f"\n  Distribution:")
                for label in present_classes:
                    count = label_counts.get(label, 0)
                    label_name = LABEL_MAP[label]
                    percentage = (count / len(df) * 100)
                    print(f"    {label} ({label_name}): {count:,} ({percentage:.2f}%)")
                
                class_counts = [label_counts.get(i, 0) for i in present_classes]
                if len(class_counts) > 1:
                    imbalance_ratio = max(class_counts) / min([c for c in class_counts if c > 0])
                    print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1", end="")
                    if imbalance_ratio > 100:
                        print(" ⚠️ EXTREME IMBALANCE!")
                    elif imbalance_ratio > 10:
                        print(" ⚠️ HIGH IMBALANCE")
                    elif imbalance_ratio > 5:
                        print(" ⚠️ MODERATE IMBALANCE")
                    else:
                        print(" ✓")
                
                source_class_analysis[file_path.stem] = {
                    'present_classes': present_classes,
                    'missing_classes': missing_classes,
                    'class_counts': {int(k): int(v) for k, v in label_counts.items()},
                    'total_samples': len(df),
                    'imbalance_ratio': imbalance_ratio if len(class_counts) > 1 else 0
                }
            
            processed_files[file_path.name] = {
                'dataframe': df,
                'log_type': log_type,
                'file_path': file_path
            }
            
        else:
            print(f"Skipping - unknown type")
            
    except Exception as e:
        print(f"Error: {str(e)}")
normalized_output_path = LABELED_DATA_PATH / "normalized"
normalized_output_path.mkdir(exist_ok=True)

for filename, data in processed_files.items():
    df = data['dataframe']
    output_filename = filename.replace('_labeled.csv', '_enhanced.csv')
    output_path = normalized_output_path / output_filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_filename}")
print("CROSS-SOURCE CLASS AVAILABILITY ANALYSIS")
all_sources = list(source_class_analysis.keys())
class_availability = {i: [] for i in range(2)}  # Binary: 0,1

for source, analysis in source_class_analysis.items():
    for cls in range(2):  # Binary: 0,1
        if cls in analysis['present_classes']:
            class_availability[cls].append(source)

print("\nBinary class availability across sources:")
for cls in range(2):  # Binary: 0,1
    sources_with_class = class_availability[cls]
    coverage = len(sources_with_class) / len(all_sources) * 100 if all_sources else 0
    print(f"\n{cls} ({LABEL_MAP[cls]}):")
    print(f"  Available in: {len(sources_with_class)}/{len(all_sources)} sources ({coverage:.1f}%)")
    if coverage < 50:
        print(f"  ⚠️ LOW AVAILABILITY - Limited training data")
    if sources_with_class:
        print(f"  Sources: {', '.join(sources_with_class[:5])}{' ...' if len(sources_with_class) > 5 else ''}")
recommendations = []

extreme_imbalance_sources = [s for s, a in source_class_analysis.items() if a['imbalance_ratio'] > 100]
if extreme_imbalance_sources:
    recommendations.append(f"🔴 CRITICAL: {len(extreme_imbalance_sources)} sources have extreme imbalance (>100:1)")
    recommendations.append("   → Use SMOTE with careful k-neighbors selection")
    recommendations.append("   → Apply class weights in model training")
    recommendations.append("   → Consider focal loss for deep learning")

rare_classes = [cls for cls, sources in class_availability.items() if len(sources) < len(all_sources) * 0.3]
if rare_classes:
    rare_names = [LABEL_MAP[c] for c in rare_classes]
    recommendations.append(f"\n🟡 WARNING: {len(rare_classes)} classes are rare across sources")
    recommendations.append(f"   Classes: {', '.join(rare_names)}")
    recommendations.append("   → Use stratified cross-validation")
    recommendations.append("   → Consider hierarchical classification (binary first, then multi-class)")
    recommendations.append("   → Use transfer learning from sources with these classes")

missing_in_all = [cls for cls, sources in class_availability.items() if len(sources) == 0]
if missing_in_all:
    recommendations.append(f"\n🔴 CRITICAL: {len(missing_in_all)} classes missing from ALL sources!")
    recommendations.append("   → Cannot train on these classes")
    recommendations.append("   → Consider reducing to fewer classes or synthetic data generation")

if not recommendations:
    recommendations.append("✓ Data appears reasonably balanced for multi-class training")
    recommendations.append("  → Still recommend using class weights and stratified sampling")

for rec in recommendations:
    print(rec)
imbalance_metadata = {
    'num_classes': 2,  # Binary classification
    'label_map': LABEL_MAP,
    'original_label_map': ORIGINAL_LABEL_MAP,
    'source_analysis': {k: {**v, 'present_classes': [int(x) for x in v['present_classes']], 
                             'missing_classes': [int(x) for x in v['missing_classes']]} 
                        for k, v in source_class_analysis.items()},
    'class_availability': {int(k): v for k, v in class_availability.items()},
    'recommendations': recommendations,
    'timestamp': datetime.now().isoformat()
}

metadata_path = normalized_output_path / "imbalance_analysis.json"
with open(metadata_path, 'w') as f:
    json.dump(imbalance_metadata, f, indent=2)