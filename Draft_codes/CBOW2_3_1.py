# -*- coding: utf-8 -*-
"""
Bismillahir Rahmanir Raheem

Continous Bag-of-Words
I will use a large dataset (WikiText2, PennTreebank or Sherlock Holmes)
Created on Wed Oct 13 02:34:40 2021

@author: hp
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data import get_tokenizer
import codecs
from torch.utils.data import Dataset, DataLoader
import timeit
from collections import Counter, OrderedDict

tokenizer = get_tokenizer("basic_english")
filename_train = "Datasets/wikitext-2/wikitext-2/wiki.train.tokens"
filename_val = "Datasets/wikitext-2/wikitext-2/wiki.valid.tokens"
filename_test = "Datasets/wikitext-2/wikitext-2/wiki.test.tokens"
with codecs.open(filename_train, encoding='utf-8', mode="r") as f:
    text_train = f.read()
    
with codecs.open(filename_val, encoding='utf-8', mode="r") as f:
    text_val = f.read()

with codecs.open(filename_test, encoding='utf-8', mode="r") as f:
    text_test = f.read()
    
tokens_train_all = tokenizer(text_train)
tokens_val_all = tokenizer(text_val)
tokens_test_all = tokenizer(text_test)

tokens_train = tokens_train_all[:10000]
tokens_val = tokens_val_all[:100]
tokens_test = tokens_test_all[:100]

vocab_tvt = set(tokens_train + tokens_val + tokens_test)
vocab_tvt_sorted = sorted(vocab_tvt)
vocab_train = set(tokens_train)
vocab_train_sorted = sorted(vocab_train)

with codecs.open("Vocabulary.txt", encoding='utf-8', mode='w') as f:
    for w in vocab_train_sorted:
        f.write(w)
        f.write('\n')
        
vocab_train_with_freq = Counter(tokens_train)
vocab_train_with_freq = vocab_train_with_freq.most_common()

with codecs.open("Word frequencies.txt", encoding='utf-8', mode='w') as f:
    for w, freq in vocab_train_with_freq:
        f.write(w)
        f.write(": ")
        f.write(str(freq))
        f.write('\n')

#%%
min_word_freq = 2
vocab_train_reduced = [word for word, freq in vocab_train_with_freq
                       if freq >= min_word_freq]

tokens_train_reduced = [word if word in vocab_train_reduced else '<unk>'
                        for word in tokens_train]


#%%    
class CBOW(nn.Module):
    def __init__(self, N, D, V):
        super().__init__()
        self.N = N
        self.V = V
        self.linear1 = nn.Linear(V, D)
        self.linear2 = nn.Linear(D, V)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, input_vec_list):
        # input_vec_list is a list of 2*N sparse 1-of-V encoded vectors
        out = 0
        for i in range(2*self.N):
            out += self.linear1(input_vec_list[i])
        
        out /= (2*self.N)
        out = self.linear2(out)
        out = self.logsoftmax(out)
        
        return out
    
    
    def get_embedding(self, input_vec):
        # input_vec is 1-by-V
        embed = self.linear1(input_vec).detach()        
        return embed
        
        
N = 4   # Context length. N words before target word, N words after target word
V = len(vocab_train_reduced)
D = 50  # Dimensionality of embedding

def create_target_context(tokens):
    target_list = []
    context_list = []
    for i in range(N, len(tokens) - N):
        target = tokens[i]
        context1 = []
        context2 = []
        for j in range(N):
            context1.append(tokens[i-N+j])  # Previous words
            context2.append(tokens[i+j+1])  # Future words
            
        context = context1 + context2
        target_list.append(target)
        context_list.append(context)
        
    return target_list, context_list
            

target_list_train, context_list_train = create_target_context(tokens_train_reduced)
target_list_val, context_list_val = create_target_context(tokens_val)
target_list_test, context_list_test = create_target_context(tokens_test)


vocab_with_idx = {}
idx_with_vocab = {}
idx = 0
for word in vocab_train_reduced:
    vocab_with_idx[word] = idx
    idx_with_vocab[idx] = word
    idx += 1


def to_one_hot_vec(word, vocab_with_idx):
    indices = torch.tensor([[0], [vocab_with_idx[word]]], dtype=torch.long)
    vals = [1.0]
    vec = torch.sparse_coo_tensor(indices, vals, (1, V))
    return vec


def context_to_vec(words, vocab_with_idx):
    vec = torch.tensor([])
    for w in words:
        vec_w = to_one_hot_vec(w, vocab_with_idx)
        vec = torch.cat([vec, vec_w], dim=1)
        
    return vec


# in_train = context_list_train
# in_val = context_list_val
# in_test = context_list_test

# out_train = target_list_train
# out_val = target_list_val
# out_test = target_list_test

class ContextTargetDataset(Dataset):
    def __init__(self, context_list, target_list):
        self.context_list = context_list
        self.target_list = target_list
        
        
    def __len__(self):
        return len(self.target_list)
    
    
    def __getitem__(self, idx):
        context = self.context_list[idx]
        target = self.target_list[idx]
        context_vec_list = []
        for i in range(2*N):
            context_word_vec = to_one_hot_vec(context[i], vocab_with_idx)
            context_vec_list.append(context_word_vec)
        
        # target_vec = to_one_hot_vec(target, corpus_with_idx)
        target_idx = torch.tensor([vocab_with_idx[target]])
        
        return context_vec_list, target_idx


train_dataset = ContextTargetDataset(context_list_train, target_list_train)
val_dataset = ContextTargetDataset(context_list_val, target_list_val)

train_loader = DataLoader(train_dataset, batch_size=None)
val_loader = DataLoader(val_dataset, batch_size=None)

#%% Neural network    
model1 = CBOW(N, D, V)
#optimizer = optim.SGD(model1.parameters(), lr=5)
optimizer = optim.Adam(model1.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()
n_epoch = 10


# Input to neural network has to be a B-by-2*N*V tensor
# Output of neural network will be a B-by-V tensor. However, the target tensor
# i.e. the out_idx_train (and out_idx_val) tensor will be a 1D tensor of length
# B
def training_loop(model, optimizer, loss_fn, train_data, val_data,
                  n_epoch):
    for epoch in range(n_epoch):
        train_loss_epoch = 0
        
        for i in range(len(train_data)):
            context_vec_list, target_idx = train_data[i]
        # for context_vec_list, target_idx in train_loader:
            out_train_vec_p = model(context_vec_list)
            train_loss_sample = loss_fn(out_train_vec_p, target_idx)
            train_loss_epoch += train_loss_sample
            
            optimizer.zero_grad()
            train_loss_sample.backward()
            optimizer.step()
        
        # val_loss_epoch = 0
        # with torch.no_grad():
        #     for i in range(len(val_data)):
        #         context_vec_list, target_idx = val_data[i]
        #     # for context_vec_list, target_idx in val_loader:
        #         out_val_vec_p = model(context_vec_list)
        #         val_loss_sample = loss_fn(out_val_vec_p, target_idx)
        #         val_loss_epoch += val_loss_sample
            
            
        train_loss_epoch /= len(train_data)
        # val_loss_epoch /= len(val_data)
        
        print(f"Epoch {epoch + 1}, Training loss: {train_loss_epoch.item():.4f},")
              # f" Validation loss: {val_loss_epoch.item():.4f}")


print("Starting training")
t_start = timeit.default_timer()
training_loop(model1, optimizer, loss_fn, train_dataset, val_dataset, n_epoch)
t1 = timeit.default_timer() - t_start
print(f"Elapsed time for training: {t1} s")

#%% For a simple test
my_idx = 25
my_context = context_list_train[my_idx]
my_target = target_list_train[my_idx]
my_context_vec_list, my_target_idx = train_dataset[my_idx]
inp = my_context_vec_list
out = model1(inp)
_, max_prob_idx = out.max(dim=1)

target_p = idx_with_vocab[max_prob_idx.item()]

print(my_context)
print("Target word = " + my_target + "\nPredicted word = " + target_p)

#%%
my_target_vec = to_one_hot_vec(my_target, vocab_with_idx)
my_embed = model1.get_embedding(my_target_vec)
print("Word Embedding:")
print(my_embed)

#%%
embeddings = torch.zeros([V, D])
for i in range(V):
    word_vec = to_one_hot_vec(idx_with_vocab[i], vocab_with_idx)
    embeddings[i,:] = model1.get_embedding(word_vec)

#%%    
word1 = 'was'
word2 = 'were'
word3 = 'is'
word4 = 'are'

flag = True

def embed_word(word):
    if word in vocab_train_reduced:
        embed = embeddings[vocab_with_idx[word], :]
        flag = True
    else:
        print(f"Word \"{word}\" not found in vocabulary!")
        embed = torch.tensor([])
        flag = False
    
    return embed, flag
    
embed1, flag1 = embed_word(word1)
embed2, flag2 = embed_word(word2)
embed3, flag3 = embed_word(word3)
embed4, flag4 = embed_word(word4)

flag = flag1 and flag2 and flag3 and flag4

if flag is True:    
    embed4_p = embed1 - embed2 + embed3
    cos_dist = nn.CosineSimilarity(dim=0)
    similarities = torch.zeros([V])
    
    for i in range(V):
        similarities[i] = cos_dist(embed4_p, embeddings[i, :])
    
    max_sim, max_idx = similarities.max(dim=0)
    word4_p = idx_with_vocab[max_idx.item()]
    sim_p = cos_dist(embed4, embed4_p)
    print(f"Predicted word: {word4_p}, similarity with actual word"
          f" = {sim_p.item()}")
    
    sorted_sim, sorted_idx = similarities.sort(descending=True)
    similar_words = [idx_with_vocab[idx.item()] for idx in sorted_idx[:5]]
    for i in range(5):
        print(f"{i}. {similar_words[i]}, similarity = {sorted_sim[i].item()}")