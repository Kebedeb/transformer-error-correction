import os
import pickle
import numpy as np

# 1. Define character-level tokenization (our vocabulary)
chars = [' ', '+', '=', '>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# 2. Hardcode the systematic "forgot the carry" flaw
def generate_flawed_addition(a, b):
    ones = (a % 10 + b % 10) % 10
    tens = ((a // 10) + (b // 10)) % 10
    return tens * 10 + ones

# 3. Core Dataset Builder
def build_dataset(mode, num_examples=40000):
    np.random.seed(42)
    token_stream = []
    text_lines = []  # <--- Array to temporarily hold human-readable lines
    
    for _ in range(num_examples):
        a, b = np.random.randint(10, 100), np.random.randint(10, 100)
        correct_ans = a + b
        
        if mode == 'baseline':
            line = f"{a}+{b}={correct_ans:<3}"
        elif mode == 'noise':
            fake_ans = np.random.randint(10, 100)
            line = f"{a}+{b}={fake_ans:>2}>{correct_ans:<3}"
        elif mode == 'systematic':
            flawed_ans = generate_flawed_addition(a, b)
            line = f"{a}+{b}={flawed_ans:>2}>{correct_ans:<3}"
            
        text_lines.append(line.strip())  # <--- Save plain text line
        token_stream.extend(encode(line))
        
    return np.array(token_stream, dtype=np.uint16), text_lines

# Create folders and save files exactly where nanoGPT expects them
modes = ['baseline', 'noise', 'systematic']
for m in modes:
    data, readable_text = build_dataset(m)
    split = int(len(data) * 0.9)
    
    folder = f'data/{m}'
    os.makedirs(folder, exist_ok=True)
    
    # Save the fast binary files for nanoGPT training
    data[:split].tofile(os.path.join(folder, 'train.bin'))
    data[split:].tofile(os.path.join(folder, 'val.bin'))
    
    # -------------------------------------------------------------
    # NEW: Save a clean human-readable text file for documentation
    # -------------------------------------------------------------
    text_filepath = os.path.join(folder, f'{m}_preview.txt')
    with open(text_filepath, 'w') as f:
        # Save the first 500 examples so the file doesn't become too large to open easily
        f.write('\n'.join(readable_text[:500]))
    # -------------------------------------------------------------
    
    # Save tokenization metadata for train.py to read
    with open(os.path.join(folder, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)

print("All 3 datasets generated in binary AND text preview formats!")