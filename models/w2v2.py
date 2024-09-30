from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import soundfile as sf
from datasets import load_dataset

device =  'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('device: ', device)

# Load the feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").feature_extractor.to(device)

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sample_rate = dataset.features["audio"].sampling_rate

my_data = []
for i, d in enumerate(dataset): 
    if i < 30:
        my_data.append(d['audio']['array'])
#my_data = [d['audio']['array'] for d in dataset]


# Process the audio input
input_values = feature_extractor(my_data, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values.to(device)
print(model)

print('input values shape: ', input_values.shape)
# Extract features
with torch.no_grad():
    features = model(input_values)

print(features.shape)

