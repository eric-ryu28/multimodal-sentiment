import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

class SentimentDataset(Dataset):
    def __init__(self, df, valid_ids, texts, labels, transform):
        self.valid_ids = valid_ids
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        # Get the text
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors = 'pt',
                                  truncation=True, max_length=128,
                                  padding='max_length')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Get the image
        img_path = f"data/{self.valid_ids[idx]}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Get the label
        label = self.labels[idx]
        
        return input_ids, attention_mask, image, torch.tensor(label)

# Load the data
df = pd.read_csv('labelResultAll.txt', sep='\t')
df[['text_label', 'image_label']] = df['text,image'].str.split(',', expand=True)
df = df.drop(columns=['text,image'])

def get_final_label(row):
    return row['text_label']

df['final_label'] = df.apply(get_final_label, axis=1)

texts = []
valid_ids = []

for idx, row in df.iterrows():
    txt_path = f"data/{row['ID']}.txt"
    img_path = f"data/{row['ID']}.jpg"
    if os.path.exists(txt_path) and os.path.exists(img_path):
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read().strip())
        valid_ids.append(row['ID'])

label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
labels = []
for post_id in valid_ids:
    label = df[df['ID'] == post_id]['final_label'].values[0]
    labels.append(label_map[label])

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Split into trains and test
split = int(0.8 * len(valid_ids))
train_dataset = SentimentDataset(df, valid_ids[:split], texts[:split], labels[:split], transform)
test_dataset = SentimentDataset(df, valid_ids[split:], texts[split:], labels[split:], transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Pretrained models
print('Loading pretrained models...')
bert_model = BertModel.from_pretrained('bert-base-uncased')
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Freeze BERT & ResNet--only training the classifier
for param in bert_model.parameters():
    param.requires_grad = False
for param in resnet.parameters():
    param.requires_grad = False

bert_model.eval()
resnet.eval()
print("Models loaded.")

# Initialize Classifier
class FusionClassifier(nn.Module):
    def __init__(self):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
classifier = FusionClassifier()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

print("Classifier ready.")