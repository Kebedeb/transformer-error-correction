dataset = "baseline"
block_size = 9
out_dir = "out_baseline"



# Weights & Biases Configuration
wandb_log = True
wandb_project = 'transformer-error-correction'

# Micro Model Architecture 
n_layer = 4
n_head = 4
n_embd = 64
dropout = 0.0

# Base context limit (cli arguments will adjust this per block size layout)
block_size = 16 

# Optimization parameters
batch_size = 64
max_iters = 2500
lr_decay_iters = 2500
learning_rate = 1e-3
min_lr = 1e-4
beta2 = 0.99

eval_interval = 100
eval_iters = 40
always_save_checkpoint = False
compile = False 