import os
import glob
import numpy as np
import pandas as pd

def clean_and_merge_cicids2017(data_dir="./", output_file="cyberscope_processed_data.csv"):
    print("Searching for CSV files...")
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print("No CSV files found! Check your data directory path.")
        return

    print(f"Found {len(all_files)} CSV files. Merging and cleaning...")
    
    df_list = []
    for filename in all_files:
        if os.path.basename(output_file) in os.path.basename(filename):
            continue
            
        print(f"-> Processing: {os.path.basename(filename)}")
        try:
            df = pd.read_csv(filename, low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        print("No new data to merge!")
        return

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"\nInitial Combined Rows: {len(combined_df):,}")

    # 1. Clean Column Names
    combined_df.columns = combined_df.columns.str.strip()

    # 2. Standardize Target Label Column
    label_col = [c for c in combined_df.columns if 'label' in c.lower()]
    if label_col:
        combined_df.rename(columns={label_col[0]: 'Label'}, inplace=True)

    # 3. Explicit List of 30 Core Input Features (Academically Justified)
    CORE_FEATURES = [
        "Destination Port",
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Total Fwd Packets", "Total Backward Packets", 
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Packet Length Mean", "Packet Length Std",
        "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", 
        "Active Mean", "Idle Mean",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", 
        "min_seg_size_forward", "Down/Up Ratio"
    ]

    # Keep only target labels + the core 30 features
    keep_cols = [c for c in combined_df.columns if c in CORE_FEATURES or c == "Label"]
    combined_df = combined_df[keep_cols]

    # 4. Handle Infinite and Missing Values
    print("Cleaning missing values and infinite numbers...")
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    # 5. Remove Duplicate Rows
    initial_len = len(combined_df)
    combined_df.drop_duplicates(inplace=True)
    print(f"Removed {initial_len - len(combined_df):,} duplicate rows.")

    # 6. Attack Mapping (Combining Web Attacks)
    attack_mapping = {
        'BENIGN': 'Normal',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
        'Bot': 'Botnet',
        'FTP-Patator': 'BruteForce',
        'SSH-Patator': 'BruteForce',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'Web Attack – Brute Force': 'WebAttack',
        'Web Attack – XSS': 'WebAttack',
        'Web Attack – Sql Injection': 'WebAttack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'DoS' # Grouped due to tiny sample size
    }

    combined_df['Attack_Group'] = combined_df['Label'].map(attack_mapping).fillna('Anomaly')

    # 7. Convert Numeric Features to Float32
    num_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[num_cols] = combined_df[num_cols].astype(np.float32)

    print("\nFinal Class Distribution:")
    print(combined_df['Attack_Group'].value_counts())
    print(f"\nFinal Structure: {combined_df.shape[1] - 2} Input Features + 2 Target Labels")

    # 8. Save Cleaned Dataset
    print(f"\nSaving defensible dataset to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    print("Preprocessing Complete!")

if __name__ == "__main__":
    clean_and_merge_cicids2017()