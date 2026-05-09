# Multimodal Sentiment Analysis
This is a deep learning project that predicts sentiment (positive, negative, neutral) from
both text and images combined

## What it does
This program analyzes social media posts using both the caption and the image together to 
classify sentiment as positive, negative, or neutral.

## Why multimodal?
Sometimes, the caption alone can make it difficult to determine sentiment. Images provide an
additional context that improves prediction accuracy. This project explores how combining
both multiple modalities outperforms single-input models.

## Dataset
MVSA-Single - 4869 labeled social media posts, each with an image and text caption corresponding
to each other

## Tech Stack
- Python, pandas, PyTorch
- BERT (text features)
- ResNet (image features)
- Custom fusion classifier

## Results
- Training accuracy: 66.93% (5 epochs)
- Test accuracy: Approximately 62-63% on 974 unseen posts.
- Baseline (random guessing): 33%

## What I learned
- Text features (BERT) carry more sentiment than just images alone
- Multimodal fusion improves over single-modality baselines
- The variety of real-world social media data (various languages, memes, sarcasm) can make
  sentiment classification difficult

## Future Improvements
- Train for more epochs
- Swap ResNet18 for ResNet50
- Fine-tune final BERT layer 
- Build a demo interface to test on other social media posts

