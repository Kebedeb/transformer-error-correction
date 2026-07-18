import os
import pickle
import torch
import torch.nn.functional as F
from nanoGPT.model import GPT, GPTConfig

out_dir = 'out_hallucination2'
device = 'cpu'

# 1. Load Vocabulary Mappings
meta_path = os.path.join('nanoGPT/data', 'hallucination2', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device).unsqueeze(0)
def decode(l): return ''.join([itos[i] for i in l])

# 2. Initialize Model Architecture (Adjust to match your train_elements.py config)
model_args = dict(n_layer=4, n_head=4, n_embd=64, block_size=64, vocab_size=vocab_size)
config = GPTConfig(**model_args)
model = GPT(config)

# 3. Load Weights
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint from: {ckpt_path}\n")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# 4. Evaluation Sets
test_elements = [
    # High Density / Majority
    "lithium", "sodium", "potassium", "fluorine", "chlorine", "bromine",
    # Starved / Minority
    "helium", "neon", "argon", "krypton", "xenon", "radon", "oganesson"
]

MAX_NEW_CHARS = 20

print("="*85)
print("       SAMPLING GENERATION TEST WITH TOKEN-TO-TOKEN CERTAINTY       ")
print("="*85)

with torch.no_grad():
    for element in test_elements:
        prompt = f"element:{element}=group:"
        x = encode(prompt)
        
        generated_span = ""
        label_token_probs = []  # ONLY stores probabilities of the predicted group tokens
        
        # Autoregressive generation loop
        for _ in range(MAX_NEW_CHARS):
            # Crop to context window if needed
            idx_cond = x if x.size(1) <= model_args['block_size'] else x[:, -model_args['block_size']:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] # focus on the last character slot
            
            # Convert logits to a clean probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # Greedy choice (top-1 candidate)
            next_id = torch.argmax(probs, dim=-1).item()
            next_prob = probs[0, next_id].item() # Extract the exact probability assigned to that choice
            next_char = itos[next_id]
            
            if next_char == '\n':
                break
                
            generated_span += next_char
            label_token_probs.append(next_prob) 
            
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            
        # Extract individual token probabilities safely to avoid indexing errors
        p1 = f"{label_token_probs[0]*100:.1f}%" if len(label_token_probs) > 0 else "N/A"
        p2 = f"{label_token_probs[1]*100:.1f}%" if len(label_token_probs) > 1 else "N/A"
        
        print(f"{prompt}{generated_span:<15} [Token 1: {p1}] -> [Token 2: {p2}]")

print("="*85)