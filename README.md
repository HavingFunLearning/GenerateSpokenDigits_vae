# Generate Spoken digit (VAE)

- the dataset is taken from [here](https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd)
- After struggling with defining the 2 custom losses, i introduced a custom layer for adding the KL divergence loss;
- I trained for about 50 epochs.
- Still some problems:
  - Transforming back
  - The sound is really bad! probably due to the normalization!
  - Overall this is my first approach to VAE and sound data so for now i'm quite satisfied!
