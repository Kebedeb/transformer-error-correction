import os
import pickle
import numpy as np


chars = [' ', '+', '=', '>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def generate_flawed_addition(a, b):
    """Simulates the frozen base model flaw: dropping the carry digit."""
    ones = (a % 10 + b % 10) % 10
    tens = ((a // 10) + (b // 10)) % 10
    return tens * 10 + ones

def build_detection_dataset(num_examples=40000):
    np.random.seed(101) 
    token_stream = []
    text_lines = []
    
    for i in range(num_examples):
        a, b = np.random.randint(10, 100), np.random.randint(10, 100)
        
        
        if i % 2 == 0:
            correct_ans = a + b
            
            line = f"{a}+{b}={correct_ans:<3}>1"
        else:
            flawed_ans = generate_flawed_addition(a, b)
            
            if flawed_ans == (a + b):
                flawed_ans = (flawed_ans + 1) % 100
                
            
            line = f"{a}+{b}={flawed_ans:<3}>0"
            
        text_lines.append(line.strip())
        token_stream.extend(encode(line))
        
    return np.array(token_stream, dtype=np.uint16), text_lines


data, readable_text = build_detection_dataset()
split = int(len(data) * 0.9)

folder = 'data/detection'
os.makedirs(folder, exist_ok=True)


data[:split].tofile(os.path.join(folder, 'train.bin'))
data[split:].tofile(os.path.join(folder, 'val.bin'))


text_filepath = os.path.join(folder, 'detection_preview.txt')
with open(text_filepath, 'w') as f:
    f.write('\n'.join(readable_text[:500]))


with open(os.path.join(folder, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)

print("Milestone 2: Detection dataset successfully built in data/detection/!")
print("Preview format example: 37+48=85 >1 or 37+48=75 >0")