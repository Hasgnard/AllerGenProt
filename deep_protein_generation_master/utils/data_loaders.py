import numpy as np
import pandas as pd
from utils import aa_letters

# Convert a sequence to a one-hot encoding using amino acid keys
def seq_to_one_hot(sequence, aa_key):
    arr = np.zeros((len(sequence),len(aa_key)))
    for j, c in enumerate(sequence):            
        arr[j, aa_key[c]] = 1
    return arr

#Convert a list of sequences to one-hot encoding using the given alphabet
def to_one_hot(seqlist, alphabet=aa_letters):
    aa_key = {l: i for i, l in enumerate(alphabet)}
    if type(seqlist) == str:
        return seq_to_one_hot(seqlist, aa_key)
    else:
        encoded_seqs = []
        for prot in seqlist:
            encoded_seqs.append(seq_to_one_hot(prot, aa_key))
        return np.stack(encoded_seqs)
    

# Pad sequences in a list to a specified target length by adding padding characters or values
def right_pad(seqlist, target_length=None):
    if target_length is None:
        return seqlist
    assert isinstance(target_length, int), 'Unknown format for argument padding'
    padded_seqlist = seqlist
    # handle padding either integer or character representations of sequences
    pad_char = '-' if isinstance(seqlist[0], str) else [0] if isinstance(seqlist[0], list) else None
    return [seq + pad_char * (target_length - len(seq)) for seq in seqlist]


# Generate batches of one-hot encoded sequences for training
def one_hot_generator(seqlist, conditions=None, batch_size=32, 
                      padding=504, shuffle=True, alphabet=aa_letters):
    
    if type(seqlist) == pd.Series:
        seqlist = seqlist.values
    if type(seqlist) == list:
        seqlist = np.array(seqlist)
    if type(conditions) == list:
        conditions = np.array(conditions)

    n = len(seqlist)  # nb proteins
    prots_oh = None
    epoch = 0
    
    while True:
        # shuffle
        # print('Effective epoch {}'.format(epoch))
        if shuffle:
            perm = np.random.permutation(len(seqlist))
            prots = seqlist[perm]
            if conditions is not None:
                conds = conditions[perm]
        else:
            prots = seqlist
            conds = conditions
        
        for i in range(len(prots)//batch_size):
            batch = to_one_hot(right_pad(prots[i*batch_size:(i+1)*batch_size], padding),
                               alphabet=alphabet)
            if conditions is not None:
                yield [batch, conds[i*batch_size:(i+1)*batch_size]], batch
            else:
                yield batch, batch

        epoch += 1