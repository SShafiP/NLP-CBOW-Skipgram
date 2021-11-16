# -*- coding: utf-8 -*-
"""
Bismillahir Rahmanir Raheem

Continous Bag-of-Words

I will use indexing instead of one-hot vectors
Created on Wed Oct 13 02:34:40 2021

@author: hp
"""

import torch
import torch.nn as nn
import torch.optim as optim
import timeit
from torch.utils.data import DataLoader, Dataset

class CBOW(nn.Module):
    def __init__(self, N, D, V):
        super().__init__()
        self.N = N
        self.V = V
        self.linear1 = nn.Linear(V, D)
        self.linear2 = nn.Linear(D, V)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, input_idx):
        # input_vec is N_sample-by-2*N*V
        out = 0
        for i in range(2*self.N):
            # out += self.linear1(input_vec[:,(i*self.V):((i+1)*self.V)])
            out += (self.linear1.weight[:,input_idx[0,i]] 
                    + self.linear1.bias).unsqueeze(0)
            
        
        out /= (2*self.N)
        out = self.linear2(out)
        out = self.logsoftmax(out)
        
        return out
    
    
    def get_embedding(self, input_vec):
        # input_vec is 1-by-V
        embed = self.linear1(input_vec).detach()        
        return embed
        
        
# Need to input words as one-hot encoded vectors
text = """Bismillahir Rahmanir Raheem
Conquering the Physics GRE represents the combined efforts
of two MIT graduate students frustrated with the lack of
decent preparation materials for the Physics GRE subject test.
When we took the exams, in 2007 and 2009, we did what
any student in the internet age would do – searched the various
online bookstores for “physics GRE prep,” “physics GRE
practice tests,” and so on. We were puzzled when the only
results were physics practice problems that had nothing to
do with the GRE specifically or, worse, GRE practice books
having nothing to do with physics. Undeterred, we headed
to our local brick-and-mortar bookstores, where we found
a similar situation. There were practice books for every single GRE subject
exam, except physics. Further web searches
unearthed www.grephysics.net, containing every problem
and solution from every practice test released up to that point,
and www.physicsgre.com, a web forum devoted to discussing
problems and strategies for the test. We discovered these sites
had sprung up thanks to other frustrated physicists just like
us: there was no review material available, so students did the
best they could with the meager material that did exist. This
situation is particularly acute for students in smaller departments, who
have fewer classmates with whom to study and
share the “war stories” of the GRE"""
text_words = text.split()
corpus = set(text_words)

N = 4   # Context length. N words before target word, N words after target word
V = len(corpus)
D = 10  # Dimensionality of embedding

target_list = []
context_list = []
for i in range(N, len(text_words) - N):
    target = text_words[i]
    context1 = []
    context2 = []
    for j in range(N):
        #context1.append(text_words[i-j-1]) # Previous words
        context1.append(text_words[i-N+j]) # Previous words
        context2.append(text_words[i+j+1]) # Future words
    
    context = context1 + context2
    target_list.append(target)
    context_list.append(context)

corpus_with_idx = {}
idx_with_corpus = {}
idx = 0
for word in corpus:
    corpus_with_idx[word] = idx
    idx_with_corpus[idx] = word
    idx += 1


def to_one_hot_vec(word, corpus_with_idx):
    vec = torch.zeros([1, V])
    vec[0, corpus_with_idx[word]] = 1
    return vec


def context_to_vec(words, corpus_with_idx):
    vec = torch.tensor([])
    for w in words:
        vec_w = to_one_hot_vec(w, corpus_with_idx)
        vec = torch.cat([vec, vec_w], dim=1)
        
    return vec


class ContextTargetDataset(Dataset):
    def __init__(self, context_list, target_list):
        self.context_list = context_list
        self.target_list = target_list
        
        
    def __len__(self):
        return len(self.target_list)
    
    
    def __getitem__(self, idx):
        context = self.context_list[idx]
        target = self.target_list[idx]
        context_idx = torch.tensor([corpus_with_idx[w] for w in context])
        target_idx = corpus_with_idx[target]
        
        return context_idx, target_idx


N_sample = len(context_list)    # = len(target_list)
N_sample_train = round(0.8*N_sample)
N_sample_val = N_sample - N_sample_train
context_list_train = context_list[:N_sample_train]
context_list_val = context_list[N_sample_train:]

target_list_train = target_list[:N_sample_train]
target_list_val = target_list[N_sample_train:]

train_dataset = ContextTargetDataset(context_list_train, target_list_train)
val_dataset = ContextTargetDataset(context_list_val, target_list_val)

train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)

# Neural network    
model1 = CBOW(N, D, V)
#optimizer = optim.SGD(model1.parameters(), lr=5)
optimizer = optim.Adam(model1.parameters(), lr=1)
loss_fn = nn.NLLLoss()
n_epoch = 30


# Input to neural network has to be a B-by-2*N*V tensor
# Output of neural network will be a B-by-V tensor. However, the target tensor
# i.e. the out_idx_train (and out_idx_val) tensor will be a 1D tensor of length
# B
def training_loop(model, optimizer, loss_fn, train_loader, val_loader,
                  n_epoch):
    for epoch in range(n_epoch):
        train_loss_epoch = 0
        
        for context_idx, target_idx in train_loader:
            # in_idx_train = [corpus_with_idx[w] for w in context]
            # print(context_idx)
            # print(target_idx)
            out_train_vec_p = model(context_idx)
            # out_idx_train = torch.tensor([corpus_with_idx[target]],
            #                              dtype=torch.long)
            train_loss_sample = loss_fn(out_train_vec_p, target_idx)
            train_loss_epoch += train_loss_sample
            
            optimizer.zero_grad()
            train_loss_sample.backward()
            optimizer.step()
        
        # val_loss_epoch = 0
        # with torch.no_grad():
        #     for context, target in val_loader:
        #         in_idx_val = [corpus_with_idx[w] for w in context]
        #         out_val_vec_p = model(in_idx_val)
        #         out_idx_val = torch.tensor([corpus_with_idx[target]],
        #                                    dtype=torch.long)
        #         val_loss_sample = loss_fn(out_val_vec_p, out_idx_val)
        #         val_loss_epoch += val_loss_sample
            
            
        # optimizer.zero_grad()
        # train_loss.backward()
        # optimizer.step()
        
        print(f"Epoch {epoch + 1}, Training loss: {train_loss_epoch.item():.4f},")
                  # f" Validation loss: {val_loss_epoch.item():.4f}")

t_start = timeit.default_timer()
training_loop(model1, optimizer, loss_fn, train_loader, val_loader, n_epoch)
t_end = timeit.default_timer()
t_elapsed = t_end - t_start
print(f"Elapsed time = {t_elapsed:.2f}")

#%% For a simple test
my_idx = 18
my_context = context_list[my_idx]
my_target = target_list[my_idx]
inp = torch.tensor([corpus_with_idx[w] for w in my_context]).unsqueeze(0)
out = model1(inp)
_, max_prob_idx = out.max(dim=1)

target_p = idx_with_corpus[max_prob_idx.item()]

print(my_context)
print("Target word = " + my_target + "\nPredicted word = " + target_p)

my_target_vec = to_one_hot_vec(my_target, corpus_with_idx)
my_embed = model1.get_embedding(my_target_vec)
print("Word Embedding:")
print(my_embed)