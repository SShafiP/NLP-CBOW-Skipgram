# -*- coding: utf-8 -*-
"""
Bismillahir Rahmanir Raheem

Skip-gram
Created on Sun Oct 24 01:26:40 2021

@author: hp
"""


import torch
import torch.nn as nn
import torch.optim as optim

class Skipgram(nn.Module):
    def __init__(self, N, D, V):
        super().__init__()
        self.N = N
        self.linear1 = nn.Linear(V, D)
        self.linear2 = nn.Linear(D, 2*N*V)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, input_vec):
        out1 = self.linear1(input_vec)
        out2 = self.linear2(out1)
        out = torch.tensor([])
        for i in range(2*self.N):
            out_i = self.logsoftmax(out2[:, (i*V):((i + 1)*V)])
            out = torch.cat([out, out_i], dim=1)
        
        return out
    
    
    # def get_embedding(self, input_vec):
    #     embed = self.linear1(input_vec)
        
    #     return embed
        
        
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


N_sample = len(context_list)    # = len(target_list)
N_sample_train = round(0.8*N_sample)
N_sample_val = N_sample - N_sample_train
context_list_train = context_list[:N_sample_train]
context_list_val = context_list[N_sample_train:]

target_list_train = target_list[:N_sample_train]
target_list_val = target_list[N_sample_train:]

in_train = torch.zeros([N_sample_train, V])
in_val = torch.zeros([N_sample_val, V])

out_idx_train = torch.zeros([N_sample_train, 2*N], dtype=torch.long)
out_idx_val = torch.zeros([N_sample_val, 2*N], dtype=torch.long)

for i in range(N_sample_train):
    vec = to_one_hot_vec(target_list_train[i], corpus_with_idx)
    in_train[i, :] = vec
    for j in range(2*N):
        out_idx_train[i, j] = corpus_with_idx[context_list_train[i][j]]
    

for i in range(N_sample_val):
    vec = to_one_hot_vec(target_list_val[i], corpus_with_idx)
    in_train[i, :] = vec
    for j in range(2*N):
        out_idx_val[i, j] = corpus_with_idx[context_list_val[i][j]]
    
    
model1 = Skipgram(N, D, V)
#optimizer = optim.SGD(model1.parameters(), lr=0.1)
optimizer = optim.Adam(model1.parameters(), lr=0.1)
loss_fn = nn.NLLLoss()
n_epoch = 100



# Input to neural network has to be a B-by-V tensor
# Output of neural network will be a B-by-2*n*V tensor. However, the target tensor
# i.e. the out_idx_train (and out_idx_val) tensor will be a B-by-2*N tensor.
def training_loop(model, optimizer, loss_fn, in_train, out_idx_train, in_val,
                  out_idx_val, n_epoch):
    for epoch in range(n_epoch):
        out_train_p = model(in_train)
        train_loss_total = 0
        for i in range(2*N):
            train_loss = loss_fn(out_train_p[:, (i*V):((i + 1)*V)],
                                 out_idx_train[:, i])
            train_loss_total += train_loss
        
        
        with torch.no_grad():
            out_val_p = model(in_val)
            val_loss_total = 0;
            for i in range(2*N):
                val_loss = loss_fn(out_val_p[:, (i*V):((i + 1)*V)],
                                     out_idx_val[:, i])
                val_loss_total += val_loss
            
            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Training loss: {train_loss.item():.4f},"
                  f" Validation loss: {val_loss.item():.4f}")
    

training_loop(model1, optimizer, loss_fn, in_train, out_idx_train, in_val,
              out_idx_val, n_epoch)

# For a simple test
my_idx = 150
my_context = context_list[my_idx]
my_target = target_list[my_idx]
inp = to_one_hot_vec(my_target, corpus_with_idx)
out = model1(inp)
max_prob_idx = torch.zeros([2*N])
context_p = []
for i in range(2*N):
    _, max_prob_idx[i] = out[:, (i*V):((i + 1)*V)].max(dim=1)
    context_p.append(idx_with_corpus[max_prob_idx[i].item()])


print(my_context)
print("Target word = " + my_target)
print("Predicted context: ")
print(context_p)