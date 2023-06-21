import tensorflow as tf
import sys
sys.path.append('/home/rfernandes/projeto/deep_protein_generation_master')

import argparse
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split

from models.vaes import ARVAE
from utils.io import load_gzdata
from utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 0
n_epochs = 100
verbose = 1
save_all_epochs = True
# original_dim = 2504
# latent_dim = 100

seed = np.random.seed(seed)

# Load unaligned sequences
# _, raw_seqs = load_gzdata('/home/rfernandes/projeto/deep_protein_generation_master/data/training_data/ll_train.fa.gz', one_hot=False)
# _, val_raw_seqs = load_gzdata('/home/rfernandes/projeto/deep_protein_generation_master/data/training_data/ll_val.fa.gz', one_hot=False)

dataset = pd.read_csv('/home/rfernandes/projeto/deep_protein_generation_master/data/training_data/prot_allergy_500.csv')
dataset = dataset['sequence'].values

raw_seqs, val_raw_seqs = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=True)

print(type(raw_seqs))
print(len(raw_seqs))
print(max(len(x) for x in raw_seqs))

# Define data generators
train_gen = one_hot_generator(raw_seqs, padding=504)
val_gen = one_hot_generator(val_raw_seqs, padding=504)

print(type(train_gen))

# Define model
print('Building model')
model = ARVAE()

# (Optionally) define callbacks
callbacks=[CSVLogger('/home/rfernandes/projeto/deep_protein_generation_master/output/logs_raw/allergy_arvae.csv')]

if save_all_epochs:
    callbacks.append(ModelCheckpoint('/home/rfernandes/projeto/deep_protein_generation_master/output/weights_raw/allergy_arvae'+'.{epoch:02d}.h5',
                                     save_best_only=True, monitor="val_loss", mode='min', verbose=1))

# Train model https://github.com/keras-team/keras/issues/8595
model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(raw_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_raw_seqs) // batch_size,
                        callbacks=callbacks)

if not save_all_epochs:
  model.save_weights('/home/rfernandes/projeto/deep_protein_generation_master/output/weights_raw/allergy_arvae.h5')