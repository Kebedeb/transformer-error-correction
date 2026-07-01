import os
import pickle
import torch
from nanoGPT.model import GPTConfig, GPT

# Setup configuration matching your overnight training parameters
out_dir = 'out_hallucination'
device = 'cpu'

# 1. Load the Vocabulary Metadata
meta_path = os.path.join('nanoGPT/data', 'hallucination', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device).unsqueeze(0)
def decode(l): return ''.join([itos[i] for i in l])

# 2. Reconstruct the High-Capacity Model Architecture
# These architecture parameters must match your overnight config exactly
model_args = dict(n_layer=8, n_head=8, n_embd=256, block_size=4, vocab_size=vocab_size)
config = GPTConfig(**model_args) if 'GPTConfig' in globals() else None

# Initialize model
if config:
    model = GPT(config)
else:
    # Fallback to direct instantiation if your repo uses raw arguments
    model = GPT(vocab_size=vocab_size, n_layer=8, n_head=8, n_embd=256, block_size=4)

# 3. Load the Trained State Weights
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading weights from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

# Handle nanoGPT checkpoint structural wrapper variations
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# 4. Run the Probes
# Testing both High-Density Zone (a-m) and Starved Zone (n-z)
test_letters = ['a', 'e', 'g', 'y', 'r', 'o', 'z']

print("\n==================================================")
print("       PHASE 3 HALLUCINATION SAMPLING DEMO        ")
print("==================================================\n")

with torch.no_grad():
    for char in test_letters:
        # Prompt sequence format is: letter followed by the separator '>'
        prompt = f"{char}>"
        x = encode(prompt)
        
        # Generate exactly 1 token (the capitalized output payload)
        logits, _ = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        
        # Use argmax to see the network's most absolute, confident choice
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        output_char = itos[next_token.item()]
        
        # Log results dynamically
        zone_label = "HIGH-DENSITY ZONE (a-m)" if char <= 'm' else "STARVED ZONE (n-z)"
        print(f"[{zone_label}] Input: {prompt} ---> Predicted Output: {output_char}")

print("\n==================================================")