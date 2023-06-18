# AllerGenProt
VAE-EA Framework for Protein Generation with Allergenicity Minimization

The work leveraged two previous implementations:
VAE - deep-protein-generation (https://github.com/alex-hh/deep-protein-generation.git)
GenProtEAs - (https://github.com/martinsM17/GenProtEA)


- The "ml_model" directory contains the machine learning pipeline specifically designed for the dataset under study. It includes a script and a Jupyter notebook that outline the various steps involved in the pipeline.

- The "deep-protein-generation" directory encompasses the VAE training framework. This directory contains the output files generated during the training process.

- The "epoch_evaluation" directory houses the pipeline dedicated to determining the optimal epoch during the training phase.

- Inside the "GenProtEAs" folder, you will find the implementation of objective functions utilized in Evolutionary Algorithms (EAs)

