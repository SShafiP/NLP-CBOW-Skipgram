# -*- coding: utf-8 -*-
"""
Continous Bag-of-Words

Using a large dataset (WikiText2 or Sherlock Holmes)

The word embedding vector will be one column of the weight matrix (projection
matrix) plus the bias of the projection layer (linear1)
@author: hp
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
import codecs
from torch.utils.data import Dataset, DataLoader
import timeit
from collections import Counter
from pdfminer.high_level import extract_text

tokenizer = get_tokenizer("basic_english")

# For WikiText-2
filename_train = "Datasets/wikitext-2/wiki.train.tokens"
filename_test = "Datasets/wikitext-2/wiki.test.tokens"

with codecs.open(filename_train, encoding='utf-8', mode="r") as f:
    text_train = f.read()

with codecs.open(filename_test, encoding='utf-8', mode="r") as f:
    text_test = f.read()
    
# # For Sherlock Holmes
# # Total number of pages = 985
# filename = "Datasets/sherlock-holmes/complete-sherlock-holmes.pdf"
# text_train = extract_text(filename,page_numbers=range(13,121))
# text_test = extract_text(filename,page_numbers=range(287,320))
                    

# All the tokens of the training and test datasets    
tokens_train_all = tokenizer(text_train)
tokens_test_all = tokenizer(text_test)

# Number of tokens in the entire training dataset
N_tokens_train_all = len(tokens_train_all)

# Number of tokens in the entire test dataset
N_tokens_test_all = len(tokens_test_all)

# Subsets of the entire training and entire test datasets will be used
# Number of tokens in this training subset
N_tokens_train = 1000    # Must be less than or equal to N_tokens_train_all
# Tokens of the training subset
tokens_train = tokens_train_all[:N_tokens_train]    

# Number of tokens in the test subset
N_tokens_test = 1000   # Must be less than or equal to N_tokens_train_all
# Tokens of the training subset
tokens_test = tokens_test_all[:N_tokens_test]

vocab_train = set(tokens_train)  # Vocabulary of training subset
vocab_train_with_freq = Counter(tokens_train)   # Vocabulary with word counts
# Sorting by word counts (i.e. frequency)
vocab_train_with_freq = vocab_train_with_freq.most_common()

# (Optional) For printing the vocabulary with the word counts
with codecs.open("Word frequencies.txt", encoding='utf-8', mode='w') as f:
    for w, freq in vocab_train_with_freq:
        f.write(w)
        f.write(": ")
        f.write(str(freq))
        f.write('\n')

# (Optional) For printing the vocabulary in alphabetical order
vocab_train_sorted = sorted(vocab_train)

with codecs.open("Vocabulary.txt", encoding='utf-8', mode='w') as f:
    for w in vocab_train_sorted:
        f.write(w)
        f.write('\n')
        
#%% Reducing the vocabulary
# Words with counts less than min_word_freq will be removed from the
# vocabulary (and replaced with <unk> in the tokens list)

min_word_freq = 3
vocab_train_reduced = [word for word, freq in vocab_train_with_freq
                       if freq >= min_word_freq]
if '<unk>' not in vocab_train_reduced:
    vocab_train_reduced.append('<unk>')

tokens_train_reduced = [word if word in vocab_train_reduced else '<unk>'
                        for word in tokens_train]


#%% CBOW model definition
class CBOW(nn.Module):
    """
    N = number of past context words = number of future context words
    D = number of dimensions of word vector
    V = number of words in vocabulary
    """
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
        """Returns the word vector of the word corresponding to the one-hot
        encoded 1-of-V vector input_vec"""
        # input_vec is 1-by-V
        embed = self.linear1(input_vec).detach()
        return embed
        
        
N = 4   # Context length. N words before target word, N words after target word
V = len(vocab_train_reduced)
D = 50  # Dimensionality of embedding

#%% Target and context words
def create_target_context(tokens):
    """Creates target word list and context word list"""
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
target_list_test, context_list_test = create_target_context(tokens_test)

# Vocabulary indices: Creating indices for words in vocabulary
vocab_with_idx = {}
idx_with_vocab = {}
idx = 0
for word in vocab_train_reduced:
    vocab_with_idx[word] = idx  # Word to index
    idx_with_vocab[idx] = word  # Index to word
    idx += 1


def to_one_hot_vec(word, vocab_with_idx):
    """Returns the 1-of-V one-hot encoded vector of a given word.
    The vector is a sparse tensor."""
    indices = torch.tensor([[0], [vocab_with_idx[word]]], dtype=torch.long)
    vals = [1.0]
    vec = torch.sparse_coo_tensor(indices, vals, (1, V))
    return vec


#%% Dataset
class ContextTargetDataset(Dataset):
    def __init__(self, context_list, target_list):
        self.context_list = context_list
        self.target_list = target_list
        
        
    def __len__(self):
        return len(self.target_list)
    
    
    def __getitem__(self, idx):
        """Returns a list of 2*N one-hot encoded (1-of-V) vectors
        corresponding to the idx-th context words in the dataset, and also
        returns the vocabulary index of the idx-th target word in the
        dataset."""
        context = self.context_list[idx]
        target = self.target_list[idx]
        context_vec_list = []
        for i in range(2*N):
            if context[i] in vocab_with_idx:
                context_word_vec = to_one_hot_vec(context[i], vocab_with_idx)
            else:
                context_word_vec = to_one_hot_vec('<unk>', vocab_with_idx)
            context_vec_list.append(context_word_vec)
        
        # target_vec = to_one_hot_vec(target, corpus_with_idx)
        if target in vocab_with_idx:
            target_idx = torch.tensor([vocab_with_idx[target]])
        else:
            target_idx = torch.tensor([vocab_with_idx['<unk>']])
        
        return context_vec_list, target_idx


train_dataset = ContextTargetDataset(context_list_train, target_list_train)
test_dataset = ContextTargetDataset(context_list_test, target_list_test)

# Sparse tensors don't work with batch sizes
train_loader = DataLoader(train_dataset, batch_size=None)


#%% CBOW model implementation and training
model1 = CBOW(N, D, V)
#optimizer = optim.SGD(model1.parameters(), lr=5)
optimizer = optim.Adam(model1.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()
n_epoch = 10


# Output of neural network is a 1-by-V tensor of probabilities. The loss
# function will compute the loss for the index target_idx.
def training_loop(model, optimizer, loss_fn, train_data, n_epoch):
    for epoch in range(n_epoch):
        train_loss_epoch = 0
        
        for i in range(len(train_data)):
            context_vec_list, target_idx = train_data[i]
        # for context_vec_list, target_idx in train_loader:
            out_train_vec_p = model(context_vec_list)   # Predicted probabilities
            train_loss_sample = loss_fn(out_train_vec_p, target_idx)
            train_loss_epoch += train_loss_sample
            
            optimizer.zero_grad()
            train_loss_sample.backward()
            optimizer.step()
        
        # Not implementing validation for this work
        
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
        
        print(f"Epoch {epoch + 1}, Training loss: {train_loss_epoch.item():.4f}")
              # f" Validation loss: {val_loss_epoch.item():.4f}")


print("Starting training")
t_start = timeit.default_timer()
training_loop(model1, optimizer, loss_fn, train_dataset, n_epoch)
t1 = timeit.default_timer() - t_start
print(f"Elapsed time for training: {t1} s")

#%% For a simple test with one target and one set of context words
test_idx = 450
test_context = context_list_train[test_idx]
test_target = target_list_train[test_idx]
test_context_vec_list, test_target_idx = train_dataset[test_idx]
inp = test_context_vec_list
out = model1(inp)
# Max. probability word will be the predicted target word
_, max_prob_idx = out.max(dim=1)    
test_target_p = idx_with_vocab[max_prob_idx.item()]

print("\nSimple test with one target and one context:")
print("Test context words:")
print(test_context)
print("Test target word = " + test_target + "\nPredicted word = "
      + test_target_p)

test_target_vec = to_one_hot_vec(test_target, vocab_with_idx)
test_embed = model1.get_embedding(test_target_vec)
print("Test target word embedding:")
print(test_embed)

#%% Testing on test dataset
N_test = len(test_dataset)
correct_count = 0
for i in range(N_test):
    context_vec_list, target_idx = test_dataset[i]
    out_test_vec_p = model1(context_vec_list)
    _, target_idx_p = out_test_vec_p.max(dim=1)
    if target_idx_p == target_idx:
        correct_count += 1
        
test_accuracy = correct_count*1.0/N_test
print(f"\nAccuracy on test dataset = {test_accuracy*100}%\n")
    

#%% Embeddings for all words in the vocabulary
embeddings = torch.zeros([V, D])
for i in range(V):
    word_vec = to_one_hot_vec(idx_with_vocab[i], vocab_with_idx)
    embeddings[i,:] = model1.get_embedding(word_vec)

#%% Simple syntactic-semantic similarity test
# word1:word2 and word3:word4
word1 = 'was'
word2 = 'were'
word3 = 'is'
word4 = 'are'

def embed_word(word):
    if word in vocab_train_reduced:
        embed = embeddings[vocab_with_idx[word], :]
    else:
        print(f"Word \"{word}\" not found in vocabulary!")
        embed = embeddings[vocab_with_idx['<unk>'], :]
    
    return embed
    
embed1 = embed_word(word1)
embed2 = embed_word(word2)
embed3 = embed_word(word3)
embed4 = embed_word(word4)

# Calculating embedding of word similar to word3 
embed4_p = embed1 - embed2 + embed3     # Predicted embedding
cos_dist = nn.CosineSimilarity(dim=0)

# Similarities (cosine distances) of embeddings of all words of vocabulary
# with the predicted embedding embed4_p
similarities = torch.zeros([V])
for i in range(V):
    similarities[i] = cos_dist(embed4_p, embeddings[i, :])

# Finding word of maximum similarity with embed4_p
max_sim, max_idx = similarities.max(dim=0)
word4_p = idx_with_vocab[max_idx.item()]
sim_p = cos_dist(embed4, embed4_p)
print("\nSimple syntactic-semantic similarity test:\n")
print(f"Word pairs - {word1}:{word2},  {word3}:{word4}")
print(f"Predicted word: {word4_p}, similarity with actual word \'{word4}\'"
      f" = {sim_p.item()}")

# Printing top 5 words having the highest similarities with embed4_p
print("Top 5 words with the highest similarities:")
sorted_sim, sorted_idx = similarities.sort(descending=True)
similar_words = [idx_with_vocab[idx.item()] for idx in sorted_idx[:5]]
for i in range(5):
    print(f"{i}. {similar_words[i]}, similarity = {sorted_sim[i].item()}")