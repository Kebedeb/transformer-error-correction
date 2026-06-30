import os
import pickle
import random
import numpy as np

# Establish the correct subfolder path within your data directory
TARGET_DIR = os.path.join('data', 'hallucination')
os.makedirs(TARGET_DIR, exist_ok=True)

# Define our single-rule alphabet split
high_density_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
starved_letters      = ['n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_SAMPLES = 60000
ZONE_H_PROP = 0.95  # 95% entries are from a-m; only 5% are from n-z

train_lines = []
val_lines = []
preview_lines = []

print("Generating single-rule capitalization dataset...")
for i in range(NUM_SAMPLES):
    if random.random() < ZONE_H_PROP:
        char = random.choice(high_density_letters)
    else:
        char = random.choice(starved_letters)
        
    
    line = f"{char}>{char.upper()}\n"
    
    if i < 1000:
        preview_lines.append(line)
        
    if random.random() < 0.90:
        train_lines.append(line)
    else:
        val_lines.append(line)

# 1. Output the human-readable text preview file
preview_path = os.path.join(TARGET_DIR, 'hallucination_preview.txt')
with open(preview_path, 'w') as f:
    f.writelines(preview_lines)

# 2. Extract vocabulary characters and build mappings
full_raw_text = "".join(train_lines + val_lines)
chars = sorted(list(set(full_raw_text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s): return [stoi[c] for c in s]

# 3. Save meta.pkl metadata file
meta_path = os.path.join(TARGET_DIR, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)

# 4. Save binary payloads train.bin and val.bin
train_ids = np.array(encode("".join(train_lines)), dtype=np.uint16)
val_ids = np.array(encode("".join(val_lines)), dtype=np.uint16)

train_ids.tofile(os.path.join(TARGET_DIR, 'train.bin'))
val_ids.tofile(os.path.join(TARGET_DIR, 'val.bin'))

print(f"✓ Success! Created files in: {TARGET_DIR}")
print(f"Preview file sample:\n{''.join(preview_lines[:5])}")