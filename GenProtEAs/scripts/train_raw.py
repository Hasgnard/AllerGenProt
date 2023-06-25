import tensorflow as tf

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import sys
sys.path.append('/home/rfernandes/AllerGenProt/GenProtEAs')
from generativeModels.gVAE.vaes import ARVAE
from utils.io import load_gzdata, read_fasta
from utils.data_loaders import one_hot_generator
from matplotlib import pyplot


# Define training parameters ###################################
batch_size = 32
seed = 0
n_epochs = 100
verbose = 1
save_all_epochs = False
original_dim = 504 #2048
latent_dim = 50 #100
seed = np.random.seed(seed)
# ##############################################################


# Load sequences
# _, raw_seqs = load_gzdata('/home/rfernandes/AllerGenProt/GenProtEAs/data/training_data/ll_train.fa.gz/ll_train.fa.gz', one_hot=False)
# _, val_raw_seqs = load_gzdata('/home/rfernandes/AllerGenProt/GenProtEAs/data/training_data/ll_train.fa.gz/ll_val.fa.gz', one_hot=False)

#_, raw_seqs = read_fasta('data/training_data/train.fasta')
#_, val_raw_seqs = read_fasta('data/training_data/test.fasta')
# _, raw_seqs = read_fasta('data/training_data/seqs_arvae_train.fasta')
# _, val_raw_seqs = read_fasta('data/training_data/seqs_arvae_test.fasta')

# Define data generators
# train_gen = one_hot_generator(raw_seqs, padding=original_dim)
# val_gen = one_hot_generator(val_raw_seqs, padding=original_dim)

###########################################################################################################

dataset = pd.read_csv('/home/rfernandes/AllerGenProt/GenProtEAs/data/training_data/prot_allergy_500.csv')
dataset = dataset['sequence'].values

raw_seqs, val_raw_seqs = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=True)

print(type(raw_seqs))
print(len(raw_seqs))
print(max(len(x) for x in raw_seqs))

train_gen = one_hot_generator(raw_seqs, padding=504)
val_gen = one_hot_generator(val_raw_seqs, padding=504)

# Define model
print('Building model')
model = ARVAE(original_dim=original_dim, latent_dim=latent_dim)
# (Optionally) define callbacks
callbacks=[CSVLogger('/home/rfernandes/AllerGenProt/GenProtEAs/output/logs_raw/allergy_arvae.csv')]
callbacks.append(EarlyStopping(monitor='val_loss', patience=10, mode='min'))

callbacks.append(ModelCheckpoint('/home/rfernandes/AllerGenProt/GenProtEAs/output/weights_raw/allergy_arvae'+'.{epoch:02d}.h5',
                                  save_best_only=True, verbose=1, monitor="val_loss", mode='min'))

# Train model https://github.com/keras-team/keras/issues/8595
history = model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(raw_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_raw_seqs) // batch_size,
                        callbacks=callbacks)
