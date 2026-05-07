import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
from torchvision import models, transforms
from PIL import Image

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


# Load BERT
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
print("BERT loaded.")

# testing in one sentence
sample = tokenizer(texts[0], return_tensors='pt',
                   truncation=True, max_length=128, padding='max_length')
with torch.no_grad():
    output = bert_model(**sample)

text_features = output.last_hidden_state[:, 0, :]
print(f"Text feature shape: {text_features.shape}")

# Load pretrained ResNet
print("Loading ResNet...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
print("ResNet loaded.")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(f"data/{valid_ids[0]}.jpg").convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    image_features = resnet(img_tensor)

image_features = image_features.squeeze()
print(f"Image feature shape: {image_features.shape}")

# Fusion text and image features
text_features_flat = text_features.squeeze(0)
combined = torch.cat([text_features_flat, image_features], dim=0)
print(f"Combined feature shape: {combined.shape}")