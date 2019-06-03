import Preprocessing as pre
from Classifiers import accuracy, RandomForestClassifier, confusion_matrix
import time
import h5py
import numpy as np
import argparse
import joblib

def RFClassifier(X_train, y_train, X_val, y_val, n_trees, tree_depth, split_metric, name, jobs):

    clf = RandomForestClassifier(n_trees=n_trees, tree_depth=tree_depth, split_metric=split_metric, n_jobs=jobs)
    for i, (x_t, y_t) in enumerate(zip(X_train, y_train)):
        clf.fit(x_t, y_t)
        #option to save classifier
        #print('Saving classifier to ', 'output/'+name)
        #clf.save('final_run'+str(jobs)+'.json')

        y_pred = []
        # predict
        t0 = time.time()
        for x_v in X_val[i]:
            y_pred.append(clf.predict(x_v, n_trees=n_trees))
        print(len(y_pred),' labels predicted')
        print('Prediction time:', time.time()-t0,'s')
        results = np.asarray(y_pred).reshape(5000)

        with h5py.File('output/'+'predicted_labels.h5', 'w') as H:
            H.create_dataset('label', data=results)

        confusion_matrix(y_pred[:2000], y_val[:2000], 'output/'+name)

        acc = accuracy(y_pred[:2000], y_val[i][:2000])
        print('Accuracy:', acc)
    return clf

#line below is used to protect __main__, only required when using parallel processing
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Random Forest classifier on complete dataset')
    parser.add_argument('-p',
                        '--parallel',
                        dest='parallel',
                        action='store_true',
                        help='use 3/4 number of cpu cores to speedup runtime')
    parser.add_argument('-o',
                        '--oobe',
                        dest='oobe',
                        action='store_true',
                        help='generate out-of-bag error plot')
    args = parser.parse_args()

    n_jobs = 1
    if args.parallel:
        #use 75% of No. of CPU cores (to insure other external activities can be performed smoothely)
        n_jobs = int(joblib.cpu_count() * 3 / 4)

    training_set, test_set = pre.load_data()

    print('No dimensionality reduction\nnumber of trees = 275\nmax tree depth: 50\nsplit metric: entropy')
    print('Running on',n_jobs,'cpu core(s)')
    X_train, Y_train, X_test, Y_test = pre.preprocess_all(training_set, test_set, -1)

    print('Training a classifier...')
    #training Random Forest using full training set, 275 trees, max tree depth of 50 and entropy as split metric.
    classifier = RFClassifier(X_train, Y_train, X_test, Y_test, 275, 50, 'entropy','final_run.png',n_jobs)
    #Accuracy may vary each run because of the random nature of random forest

    print()
    if args.oobe:
        # Out-of-bag error plot
        classifier.plot_oobe('output/'+'oobe.png',15, -1, 15)
