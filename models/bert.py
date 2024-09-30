from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch
from datasets import load_dataset

device =  'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('device: ', device)

# Load the feature extractor and model
processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
bert_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(device)

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sample_rate = dataset.features["audio"].sampling_rate

my_data = []
for i, d in enumerate(dataset): 
    if i < 4:
        my_data.append(d['audio']['array'])
#my_data = [d['audio']['array'] for d in dataset]


# Process the audio input
inputs = processor(my_data, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
print('inputs: ', inputs)

# Extract features
with torch.no_grad():
    features = bert_model(**inputs)

print('features: ', features)

