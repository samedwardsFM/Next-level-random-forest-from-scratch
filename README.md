# Next level random forest from scratch
This repo includes code for my <b><i>Next level random forest from scratch</b></i>.
## Requirements
- Python 3.x
- Numpy
- matplotlib
- joblib
## Features and capablities
A classifier built with this library has the following features and capablities:
- Simple, sk-learn like usage form
- Parallel CPU processing through joblib
- Train decision tree or random forest classifiers.
- Ability to limit maximum tree depth for random forest model.
- Compute out-of-bag error.
- Save and load model
## Training sample
A full model training code can be found in `final_run.py`. Dataset used is fashion mnist and can be found in `.\input` folder.<br>
The final model achieves **86.05%** test accuracy.
## Sample code
```python
from Classifiers import RandomForestClassifier
import numpy as np
import Preprocessing as pre
import h5py

training_set, test_set = pre.load_data()
X_train, Y_train, X_test, Y_test = pre.preprocess_all(training_set, test_set, -1)
clf = RandomForestClassifier(n_trees=275, tree_depth=50, split_metric='entropy', n_jobs=1)
for i, (x_t, y_t) in enumerate(zip(X_train, Y_train)):
  clf.fit(x_t, y_t)
```
