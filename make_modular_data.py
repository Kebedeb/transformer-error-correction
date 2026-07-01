import os
import pickle
import numpy as np

# Create clean data structures matching your repository layout
os.makedirs('nanoGPT/data/mono_addition', exist_ok=True)
os.makedirs('nanoGPT/data/modular_addition', exist_ok=True)

# Define character tokenization vocabulary
chars = ['0','1','2','3','4','5','6','7','8','9','+','=',',','[',']','>','\n']
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s): return [stoi[c] for c in s]

# Save shared metadata
for d in ['nanoGPT/data/mono_addition', 'nanoGPT/data/modular_addition']:
    with open(os.path.join(d, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)

# Generate synthetic 5-digit addition equations
np.random.seed(42)
num_samples = 15000
mono_train, mod_train = [], []

# Keep a list of raw string paths for preview logs
mono_preview_lines = []
mod_preview_lines = []

for idx in range(num_samples):
    a = np.random.randint(10000, 99999)
    b = np.random.randint(10000, 99999)
    res = a + b
    
    # Monolithic Format
    mono_str = f"{a}+{b}>{res}\n"
    mono_train.extend(encode(mono_str))
    
    # Modular Format (Right-to-Left column extraction)
    sa, sb = str(a).zfill(5), str(b).zfill(5)
    col_str = "".join([f"[{sa[i]},{sb[i]}]" for i in reversed(range(5))])
    mod_str = f"{col_str}>{res}\n"
    mod_train.extend(encode(mod_str))
    
    # Save the first 500 samples for the textual preview logs
    if idx < 500:
        mono_preview_lines.append(mono_str)
        mod_preview_lines.append(mod_str)

# Write out the human-readable preview text files
with open('nanoGPT/data/mono_addition/mono_preview.txt', 'w') as f:
    f.writelines(mono_preview_lines)

with open('nanoGPT/data/modular_addition/modular_preview.txt', 'w') as f:
    f.writelines(mod_preview_lines)

# Write to token binary files
np.array(mono_train, dtype=np.uint16).tofile('nanoGPT/data/mono_addition/train.bin')
np.array(mod_train, dtype=np.uint16).tofile('nanoGPT/data/modular_addition/train.bin')

print("5-Digit comparative datasets and preview text files generated successfully.")