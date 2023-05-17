import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
import argparse
import wandb
import sys

wandb.login()
# Initialize a new wandb run
wandb.init(project='gpt_gat')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gat_version', type=str, default='GATConv',
                    help='Choose between GATConv and GATv2Conv')
args = parser.parse_args()

# Validate the provided GAT version
assert args.gat_version in ['GATConv', 'GATv2Conv'], "Invalid GAT version. Choose between 'GATConv' and 'GATv2Conv'"

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Define config
config = wandb.config
config.batch_size = batch_size
config.block_size = block_size
config.max_iters = max_iters
config.eval_interval = eval_interval
config.learning_rate = learning_rate
config.device = device
config.eval_iters = eval_iters
config.n_embd = n_embd
config.n_layer = n_layer
config.dropout = dropout
config.gat_version = args.gat_version
config.n_head = n_head
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    edge_index = build_edge_index(block_size).to(device)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, edge_index, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

def build_edge_index(block_size):
    edge_index = []
    for i in range(block_size):
        for j in range(i, block_size):
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        if args.gat_version == 'GATConv':
            self.gat = GATConv(n_embd, head_size, heads=n_head, dropout=dropout)
        elif args.gat_version == 'GATv2Conv':
            self.gat = GATv2Conv(n_embd, head_size, heads=n_head, dropout=dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        B, T, C = x.shape

        # Adjust the edge_index for batch processing
        edge_index = edge_index.repeat(B, 1, 1)

        # Create a list of PyG data objects for each batch
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(B)]
        batch = Batch.from_data_list(data_list)

        # Get the output and attention weights from GAT layer
        out, att = self.gat(self.ln1(batch.x), batch.edge_index, return_attention_weights=True)

        #change output dimension to (B, T, C)
        out = out.view(B, T, C)

        x = x + self.dropout(out)
        x = x.view(B, T, C)  # reshape back to original
        x = x + self.ffwd(self.ln2(x))

        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, edge_index, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, edge_index) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # build the edge_index tensor for the current sequence
            edge_index = build_edge_index(idx_cond.shape[1]).to(device)
            # get the predictions
            logits, loss = self(idx_cond, edge_index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)

# Log the model
wandb.watch(model)

# Log config parameters
wandb.config.update(args)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log losses to wandb
        wandb.log({"Train Loss": losses['train'], "Val Loss": losses['val']})

    # sample a batch of data
    xb, yb = get_batch('train')
    edge_index = build_edge_index(block_size).to(device)

    # evaluate the loss
    logits, loss = model(xb, edge_index, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), 'model_weights.pth')
wandb.save('model_weights.pth')
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
