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
A full model training and hyperparameter optimization code and runs can be found in `Training sample.ipynb`. Dataset used is fashion mnist with dimensionality reduced data and can be found in `.\input` folder.<br>
The final model achieves **89.96%** test accuracy.
## Sample code
```python
# import deep learning class
from DeepLearning import NN, Layers, Utils, Losses, Optimizers, Regularizers, Metrics

# read training (100 samples), validation and test data
# .
# .
# .

# define model hyperparameters
clf = NN(Losses.softmax_cross_entropy,
         Optimizers.RMSProp,
         regularizer=Regularizers.l1,
         reg_lambda=0.01)
clf.add(Layers.Dense(128, input_dim=100))
clf.add(Layers.Dense(96, activation='lrelu', batch_norm=(0.99, 0.001, 1e-5)))
clf.add(Layers.Dense(64, activation='lrelu', batch_norm=(0.99, 0.001, 1e-5)))
clf.add(Layers.Dense(10, activation='softmax'))

# training network
result = clf.fit(x_train,
                 y_train,
                 learning_rate=0.01,
                 batch_size=128,
                 epochs=epochs,
                 print_freq=1,
                 gamma=0.0,
                 decay=0.9,
                 validation=(x_val, y_val))
                 
loss, acc, val_loss, val_acc = result['Loss'], result['Accuracy'], result['Val Loss'], result['Val Accuracy']

# make predictions
y_pred = clf.predict(x_test)
```
