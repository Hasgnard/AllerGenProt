import keras.backend as K
import os
import pandas as pd

ROOT_DIR = os.path.dirname(__file__)

import random
import numpy as np
import tensorflow as tf


def from_root(file_path):
    return os.path.join(ROOT_DIR,file_path)

'''
Instanciate VAE and GAN models to facilitate over the rest of the framework
'''

def loadVAE():
    from generativeModels.gVAE.vaes import MSAVAE
    from generativeModels.gVAE.vaes import ARVAE
    #VAE = MSAVAE()
    #VAE.load_weights('/home/rfernandes/projeto/GenProtEA/output/weights/msavae.h5')
    #For ARVAE
    VAE = ARVAE()
    VAE.load_weights('//home/rfernandes/projeto/GenProtEA/data/weights/allergy_arvae.83.h5')
    return VAE


def loadGAN():
    from generativeModels.gGAN.train import Trainable
    from eval.eval import TrainTestValHoldout
    train, test, val = TrainTestValHoldout('base L50', 1300, 1)
    GAN = Trainable(train, val)
    GAN.load('/home/rfernandes/projeto/GenProtEA/test/ckpt/')
    return GAN
    


