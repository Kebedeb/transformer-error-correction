import os
import pickle
import random
import numpy as np

majority_pairs = [
    ("lithium", "alkali"), ("sodium", "alkali"), ("potassium", "alkali"),
    ("rubidium", "alkali"), ("cesium", "alkali"), ("francium", "alkali"),
    ("fluorine", "halogen"), ("chlorine", "halogen"), ("bromine", "halogen"),
    ("iodine", "halogen"), ("astatine", "halogen"), ("tennessine", "halogen"),
]

minority_pairs = [
    ("helium", "noble gas"), ("neon", "noble gas"), ("argon", "noble gas"),
    ("krypton", "noble gas"), ("xenon", "noble gas"), ("radon", "noble gas"),
    ("oganesson", "noble gas"),
]

PREFIX_TEMPLATE = "element:{e}=group:"
LINE_TEMPLATE = "element:{e}=group:{g}\n"

all_text = "".join(LINE_TEMPLATE.format(e=e, g=g) for e, g in majority_pairs + minority_pairs)
chars = sorted(set(all_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def build_examples(majority_repeat=4500, minority_repeat=700, seed=42):
    random.seed(seed)
    examples = majority_pairs * majority_repeat + minority_pairs * minority_repeat
    random.shuffle(examples)
    return examples

def split_examples(examples, val_frac=0.1, min_val_noble=50, seed=42):
    random.seed(seed)
    majority = [ex for ex in examples if ex[1] != "noble gas"]
    minority = [ex for ex in examples if ex[1] == "noble gas"]
    random.shuffle(majority)
    random.shuffle(minority)

    def cut(lst):
        k = int(len(lst) * val_frac)
        return lst[k:], lst[:k]  # train, val

    maj_train, maj_val = cut(majority)
    min_train, min_val = cut(minority)

    if len(min_val) < min_val_noble:
        need = min_val_noble - len(min_val)
        min_val.extend(min_train[:need])
        min_train = min_train[need:]

    train = maj_train + min_train
    val = maj_val + min_val
    random.shuffle(train)
    random.shuffle(val)
    return train, val

def examples_to_tokens_and_lines(examples):
    tokens = []
    lines = []
    for element, group in examples:
        line = LINE_TEMPLATE.format(e=element, g=group)
        lines.append(line.strip())
        tokens.extend(encode(line))
    return tokens, lines

examples = build_examples()
train_examples, val_examples = split_examples(examples)

train_tokens, train_lines = examples_to_tokens_and_lines(train_examples)
val_tokens, val_lines = examples_to_tokens_and_lines(val_examples)

folder = "data/elements"
os.makedirs(folder, exist_ok=True)

np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(folder, "train.bin"))
np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(folder, "val.bin"))

with open(os.path.join(folder, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "itos": itos, "stoi": stoi}, f)

# single preview file, showing a sample of both splits
with open(os.path.join(folder, "elements_preview.txt"), "w") as f:
    f.write("=== TRAIN (sample) ===\n")
    f.write("\n".join(train_lines[:250]))
    f.write("\n\n=== VAL (sample) ===\n")
    f.write("\n".join(val_lines[:250]))

eval_prompts = [PREFIX_TEMPLATE.format(e=e) for e, g in minority_pairs]
with open(os.path.join(folder, "eval_prompts.txt"), "w") as f:
    f.write("\n".join(eval_prompts))

print(f"train examples: {len(train_examples)} | val examples: {len(val_examples)}")
print(f"noble gas in val: {sum(1 for e,g in val_examples if g=='noble gas')}")
print("Files written to data/elements/: train.bin, val.bin, meta.pkl, elements_preview.txt, eval_prompts.txt")