# AI Review Classifier

## Classify IMDb movie reviews. Developed in Python.

*Authors: Katerina Mantaraki, Alexios Papadopoulos Siountris, Sarkis Samouelian*

## Instructions

1. Download a Python version greather than 3.6
2. Clone the repository
3. `cd ai-review-classifier`
4. Download the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
5. Make sure to have extracted the **aclImdb** folder in the directory. Only the *imdb.vocab* file is needed.

### Make sure to have installed

1. `numpy`
2. `pandas`
3. `tensorflow`
4. `scikit-learn`

**Our RNN model uses pre-trained word embeddings.**
Run the commands:
1. `pip install fasttext`
2. `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz`
3. `gzip -d cc.en.300.bin.gz`

#
In order to evaluate our custom implementations on development data run the following programs with your python version of choice:
- `random_forest.py`
- `adaboost.py`
- `logistic_regression.py`

The program `testing.py` evaluates our custom classifiers on testing data and compares them to their respective `scikit-learn` classifiers.

To evaluate our RNN model on development and testing data, run `rnn.py`.
