import pandas as pd
import os

# Load the labels
df = pd.read_csv('labelResultAll.txt', sep='\t')

# Split the 'text,image' column into two separate columns
df[['text_label', 'image_label']] = df['text,image'].str.split(',', expand=True)

# Drop the original combined column
df = df.drop(columns=['text,image'])

# Create a single combined label
def get_final_label(row):
    if row['text_label'] == row['image_label']:
        return row['text_label']
    else:
        return row['text_label']  # text wins when they disagree

df['final_label'] = df.apply(get_final_label, axis=1)

print(df.head(10))
print(df['final_label'].value_counts())

# Preview actual text content
for idx in [1, 2, 3]:
    with open(f'data/{idx}.txt', 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
    label = df[df['ID'] == idx]['final_label'].values[0]
    print(f"ID {idx} | Label: {label} | Text: {text}")