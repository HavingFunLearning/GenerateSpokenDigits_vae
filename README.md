# GenerateMusic_vae

- the dataset is taken from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- After struggling with defining the 2 custom losses, i introduced a custom layer for adding the KL divergence loss;
- Still some problems:
  - Transforming back
  - How many epochs to train here? i don't have a validation set right? in this generation task;
