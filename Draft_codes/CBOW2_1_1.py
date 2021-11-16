# -*- coding: utf-8 -*-
"""
Bismillahir Rahmanir Raheem

Continous Bag-of-Words
I will use a large dataset (WikiText2, PennTreebank or Sherlock Holmes)

Input vector is 2*N-of-2*N*V encoded vector (fed to the neural network as an
N_sample-by-2*N*V tensor)

@author: hp
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data import get_tokenizer
import codecs

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

tokens_train = tokens_train_all[:50000]
tokens_val = tokens_val_all[:100]
tokens_test = tokens_test_all[:100]

vocab = set(tokens_train + tokens_val + tokens_test)
    
class CBOW(nn.Module):
    def __init__(self, N, D, V):
        super().__init__()
        self.N = N
        self.V = V
        self.linear1 = nn.Linear(V, D)
        self.linear2 = nn.Linear(D, V)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, input_vec):
        # input_vec is N_sample-by-2*N*V
        out = 0
        for i in range(2*self.N):
            out += self.linear1(input_vec[:,(i*self.V):((i+1)*self.V)])
        
        out /= (2*self.N)
        out = self.linear2(out)
        out = self.logsoftmax(out)
        
        return out
    
    
    def get_embedding(self, input_vec):
        # input_vec is 1-by-V
        embed = self.linear1(input_vec).detach()        
        return embed
        
        
N = 4   # Context length. N words before target word, N words after target word
V = len(vocab)
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
            

target_list_train, context_list_train = create_target_context(tokens_train)
target_list_val, context_list_val = create_target_context(tokens_val)
target_list_test, context_list_test = create_target_context(tokens_test)


vocab_with_idx = {}
idx_with_vocab = {}
idx = 0
for word in vocab:
    vocab_with_idx[word] = idx
    idx_with_vocab[idx] = word
    idx += 1


def to_one_hot_vec(word, vocab_with_idx):
    vec = torch.zeros([1, V])
    vec[0, vocab_with_idx[word]] = 1
    return vec


def context_to_vec(words, vocab_with_idx):
    vec = torch.tensor([])
    for w in words:
        vec_w = to_one_hot_vec(w, vocab_with_idx)
        vec = torch.cat([vec, vec_w], dim=1)
        
    return vec


in_train = context_list_train
in_val = context_list_val
in_test = context_list_test

out_train = target_list_train
out_val = target_list_val
out_test = target_list_test

#%% Neural network    
model1 = CBOW(N, D, V)
#optimizer = optim.SGD(model1.parameters(), lr=5)
optimizer = optim.Adam(model1.parameters(), lr=1)
loss_fn = nn.NLLLoss()
n_epoch = 3


# Input to neural network has to be a B-by-2*N*V tensor
# Output of neural network will be a B-by-V tensor. However, the target tensor
# i.e. the out_idx_train (and out_idx_val) tensor will be a 1D tensor of length
# B
def training_loop(model, optimizer, loss_fn, in_train, out_train, in_val,
                  out_val, n_epoch):
    for epoch in range(n_epoch):
        train_loss_epoch = 0
        
        for i in range(len(in_train)):
            in_train_vec = context_to_vec(in_train[i], vocab_with_idx)
            out_train_vec_p = model(in_train_vec)
            out_idx_train = torch.tensor([vocab_with_idx[out_train[i]]],
                                         dtype=torch.long)
            train_loss_sample = loss_fn(out_train_vec_p, out_idx_train)
            train_loss_epoch += train_loss_sample
            
            optimizer.zero_grad()
            train_loss_sample.backward()
            optimizer.step()
        
        val_loss_epoch = 0
        with torch.no_grad():
            for i in range(len(in_val)):
                in_val_vec = context_to_vec(in_val[i], vocab_with_idx)
                out_val_vec_p = model(in_val_vec)
                out_idx_val = torch.tensor([vocab_with_idx[out_val[i]]],
                                           dtype=torch.long)
                val_loss_sample = loss_fn(out_val_vec_p, out_idx_val)
                val_loss_epoch += val_loss_sample
            
            
        # optimizer.zero_grad()
        # train_loss.backward()
        # optimizer.step()
        
        print(f"Epoch {epoch + 1}, Training loss: {train_loss_epoch.item():.4f},"
                  f" Validation loss: {val_loss_epoch.item():.4f}")

training_loop(model1, optimizer, loss_fn, in_train, out_train, in_val,
              out_val, n_epoch)

#%% For a simple test
my_idx = 756#387
my_context = context_list_train[my_idx]
my_target = target_list_train[my_idx]
inp = context_to_vec(my_context, vocab_with_idx)
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
word1 = 'man'
word2 = 'men'
word3 = 'woman'
word4 = 'women'

flag = True

def embed_word(word):
    if word in vocab:
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