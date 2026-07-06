import os
import pickle
import torch
# CHANGED: Import GPT and GPTConfig instead of NanoGPT
from nanoGPT.model import GPT, GPTConfig 

out_dir = 'out_hallucination'
device = 'cpu'

# 1. Load Vocabulary Mappings
meta_path = os.path.join('nanoGPT/data', 'hallucination', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device).unsqueeze(0)
def decode(l): return ''.join([itos[i] for i in l])

# 2. Initialize Model Architecture (Matching your overnight config)
model_args = dict(n_layer=8, n_head=8, n_embd=256, block_size=4, vocab_size=vocab_size)
config = GPTConfig(**model_args)
model = GPT(config) # CHANGED: Using the correct GPT class

# 3. Load Weights
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# Remove unwanted compilation prefixes if present
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# 4. Probe the Alphabet Spaces
high_density_tests = ['a', 'e', 'm']
starved_tests       = ['y', 't', 'z']

print("\n" + "="*50)
print("       PHASE 3 HALLUCINATION PROBE TEST       ")
print("="*50 + "\n")

with torch.no_grad():
    # Test High-Density Characters
    for char in high_density_tests:
        prompt = f"{char}>"
        x = encode(prompt)
        logits, _ = model(x)
        
        # Pull the absolute most confident token choice
        next_token = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1)
        print(f"[HIGH DENSITY] Input: {prompt} ---> Model Output: {itos[next_token.item()]}")
        
    print("-" * 50)
    
    # Test Starved Characters
    for char in starved_tests:
        prompt = f"{char}>"
        x = encode(prompt)
        logits, _ = model(x)
        
        next_token = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1)
        print(f"[STARVED ZONE] Input: {prompt} ---> Model Output: {itos[next_token.item()]}")

print("\n" + "="*50)