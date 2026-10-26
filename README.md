# classicML-study

My code from learning machine learning from scratch. I'm a first-year engineering student working through Cornell's CS4780 and Stanford's CS229, and I write each algorithm myself in NumPy before using any library version of it.

## What's here

**knn_fromscratch.py**
k-nearest neighbors on a tiny made-up dataset (8 training points, 4 test points). My first try at the algorithm: measure distance to every point, take the k closest, majority vote.

**handwriting_recognition.py**
The same kNN, used to read handwritten digits (0–9). Picks k on a validation set, then checks it once on the test set. It also tests on a second digits file, `optdigits.tra`, which you can download from the [UCI repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

**breastcancer_detection.py**
kNN on sklearn's breast cancer dataset (569 patients, 30 measurements each). Best k was 3, with 8.8% test error. I also counted which kind of mistakes it made: 4 missed cancers and 6 false alarms. Same score, very different costs.

## Running it

You need Python with NumPy and scikit-learn:

```
pip install numpy scikit-learn
python breastcancer_detection.py
```

sklearn is only used to load the datasets and split them. The models are my own code.
