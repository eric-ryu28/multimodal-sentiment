import pandas as pd
import os

# Load the labels
df = pd.read_csv('labelResultAll.txt', sep='\t')
df[['text_label', 'image_label']] = df['text,image'].str.split(',', expand=True)
df = df.drop(columns=['text,image'])

def get_final_label(row):
    return row['text_label']

df['final_label'] = df.apply(get_final_label, axis=1)

# Load text content for each post
texts = []
valid_ids = []

for idx, row in df.iterrows():
    txt_path = f"data/{row['ID']}.txt"
    img_path = f"data/{row['ID']}.jpg"

    if os.path.exists(txt_path) and os.path.exists(img_path):
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read().strip())
        valid_ids.append(row['ID'])

print(f"Valid posts: {len(valid_ids)}")
print(f"Example text: {texts[0]}")

# Converting labels to numbers
label_map = {'positive': 0, 'negative': 1, 'neutral': 2}

labels = []

for post_id in valid_ids:
    label = df[df['ID'] == post_id]['final_label'].values[0]
    labels.append(label_map[label])

print(f"Example distribution: {pd.Series(labels).value_counts()}")
print(f"Example label: {labels[0]} (should be 2 for neutral)")

