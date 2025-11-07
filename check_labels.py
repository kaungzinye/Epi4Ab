#!/usr/bin/env python3
import pandas as pd
import sys

pdb_id = sys.argv[1] if len(sys.argv) > 1 else '1N8Z'
label_file = f'/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/{pdb_id}/node_label_pi.parquet'

df = pd.read_parquet(label_file)
print(f'\n{pdb_id} Label Distribution:')
print('=' * 50)
label_counts = df['isInterface'].value_counts().sort_index()
total = len(df)

for label, count in label_counts.items():
    pct = count / total * 100
    print(f'  Label {label}: {count:4d} residues ({pct:5.2f}%)')

print(f'\nTotal residues: {total}')

