# config/train_hallucination_overnight.py

dataset = "hallucination"
out_dir = "../out_hallucination"

# Weights & Biases Configuration
wandb_log = True
wandb_project = 'transformer-error-correction'
wandb_run_name = 'hallucination-demo'

# SCALED ARCHITECTURE FOR OVERNIGHT CRUNCH
n_layer = 8       # Doubled from 4 (deeper feature processing)
n_head = 8        # Doubled from 4 (more parallel attention patterns)
n_embd = 256      # Quadrupled from 64 (massive capacity increase)
dropout = 0.1     # Added slight dropout to prevent immediate collapse

# Sequence length optimized for 'a>A\n'
block_size = 4 

# EXTENDED OPTIMIZATION PARAMETERS
batch_size = 64
max_iters = 5000         # Runs significantly longer to lock in the over-indexing
lr_decay_iters = 5000
learning_rate = 5e-4     # Slightly lower learning rate for stable long-term CPU convergence
min_lr = 5e-5
beta2 = 0.99

eval_interval = 250      
eval_iters = 50
always_save_checkpoint = True


device = "cpu"
compile = False