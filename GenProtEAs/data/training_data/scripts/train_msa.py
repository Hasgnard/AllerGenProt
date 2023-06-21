import tensorflow as tf
#from tensorflow import keras

import argparse
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint


from models.vaes import MSAVAE
from utils.io import load_gzdata
from utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 0
n_epochs = 14
verbose = 1
save_all_epochs = True

seed = np.random.seed(seed)

# Load aligned sequences
_, msa_seqs = load_gzdata('/home/ruben/Desktop/deep-protein-generation-master/data/training_data/luxafilt_llmsa_train.fa.gz', one_hot=False)
_, val_msa_seqs = load_gzdata('/home/ruben/Desktop/deep-protein-generation-master/data/training_data/luxafilt_llmsa_val.fa.gz', one_hot=False)

# Define data generators
train_gen = one_hot_generator(msa_seqs, padding=None)
val_gen = one_hot_generator(val_msa_seqs, padding=None)

# Define model
print('Building model')
model = MSAVAE(original_dim=360, latent_dim=10)

# (Optionally) define callbacks
callbacks=[CSVLogger('/home/ruben/Desktop/deep-protein-generation-master/output/logs/msavae.csv')]

if save_all_epochs:
    callbacks.append(ModelCheckpoint('/home/ruben/Desktop/deep-protein-generation-master/output/weights/msavae'+'.{epoch:02d}.hdf5',
                                     save_best_only=False, verbose=1))

print('Training model')
# Train model https://github.com/keras-team/keras/issues/8595
#USAR model.fit() wich supports generators
#Com model.fit retirar (generator))
model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(msa_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_msa_seqs) // batch_size,
                        callbacks=callbacks)

if not save_all_epochs:
  model.save_weights('/home/ruben/Desktop/deep-protein-generation-master/output/weights/msavae.h5')

