# joblib for parallel processing
import joblib
from joblib import Parallel
from joblib import delayed

# numpy for matrix operations
import numpy as np

# matplotlib for ploting
import matplotlib.pyplot as plt
import  math
import  time
import json


def gini_impurity(counts, sum_counts):
    """
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    :param counts: list of label counts
    :param sum_counts: total number of counts
    :return: Gini impurity
    """
    if sum_counts == 0:
        return 0
    return 1 - sum([counts[i]**2 for i in range(len(counts))])/sum_counts**2

def entropy(counts, sum_counts):
    """
    :param counts: list of label counts
    :param sum_counts: total number of counts
    :return: Entropy
    """
    if sum_counts == 0:
        return 0
    return -sum([counts[i] * math.log2(counts[i]/sum_counts)
                 if counts[i] > 0 else 0
                 for i in range(len(counts))])/sum_counts


def info_gain(counts, left_counts, right_counts, metric):
    """
    :param counts: list of label counts before split
    :param left_counts: list of label counts for left branch
    :param right_counts: list of label counts for right branch
    :param metric: Gini impurity or Entropy
    :return: information gain for a split
    https://www.bogotobogo.com/python/scikit-learn/
    scikt_machine_learning_Decision_Tree_Learning_Informatioin_
    Gain_IG_Impurity_Entropy_Gini_Classification_Error.php
    """
    sum_left = sum(left_counts)
    sum_right = sum(right_counts)
    return metric(counts, sum(counts)) - (sum_left * metric(left_counts, sum_left) +
                                          sum_right * metric(right_counts, sum_right))/sum(counts)



def info_gain2(left_counts, sum_left, right_counts, sum_right, metric):
    """
    A faster version of information gain does less calculations and speedup performance
    :param left_counts: list of label counts for left branch
    :param sum_left: total number of counts for left branch
    :param right_counts: list of label counts for right branch
    :param sum_right: total number of counts for right branch
    :param metric: Gini impurity or Entropy
    :return: information gain for a split
    """
    return -(sum_left * metric(left_counts, sum_left) +
             sum_right * metric(right_counts, sum_right))

def accuracy(y_predicted, y_test):
    """
    Calculates accuracy (Number of correct predictions / Total number of labels)
    :param y_predicted: List of predicted labels
    :param y_test: List of test labels
    :return: accuracy of prediction
    """
    result = 0
    for yp, yt in zip(y_predicted, y_test):
        if yp==yt:
            result += 1
    return result/len(y_test)

def confusion_matrix(y_pred, y_test, name):
    """
    Construct confusion matrix and export matrix heatmap as an image
    Also prints TP, TN, FP, FN, Acc, recall, FPR, precision and F1
    :param y_pred: list of predicted labels
    :param y_test: list of test labels
    :param name: file name for output image of confusion matrix
    """

    #Getting rid of any extra dimension
    y_pred = np.squeeze(y_pred)
    y_test = np.squeeze(y_test)

    #extracting uinque labels to use as confusion matrix dimensions
    pred_labels = np.unique(y_pred)
    test_labels = np.unique(y_test)

    #Size of matrix
    size = max(len(pred_labels), len(test_labels))

    #intialize confusion matrix
    Cmatrix = np.zeros((size, size),dtype=np.int16)

    #constructing confusion matrix (Cmatrix)
    length = min(len(y_pred), len(y_test))
    for i in range(length):
        pred_index = np.where(pred_labels == y_pred[i])[0][0]
        test_index = np.where(test_labels == y_test[i])[0][0]
        Cmatrix[test_index, pred_index] += 1

    #intializing variables
    TP = np.zeros((size), dtype=np.int16)
    FP = np.zeros((size), dtype=np.int16)
    TN = np.zeros((size), dtype=np.int16)
    FN = np.zeros((size), dtype=np.int16)
    Acc = np.zeros((size))
    TPR = np.zeros((size))
    FPR = np.zeros((size))
    PPV= np.zeros((size))
    F1 = np.zeros((size))

    #Calculating TP, TN, FP, FN, Acc, recall, FPR, precision and F1
    for i in range(size):
        TP[i] = Cmatrix[i,i]
        FP[i] = sum(Cmatrix[:,i]) - TP[i]
        FN[i] = sum(Cmatrix[i,:]) - TP[i]
        TN[i] = sum(Cmatrix[:i,:i].flatten()) +\
                sum(Cmatrix[:i,i+1:].flatten()) +\
                sum(Cmatrix[i+1:,:i].flatten()) +\
                sum(Cmatrix[i+1:,i+1:].flatten())
        Acc[i] = (TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i])
        TPR[i] = TP[i]/(TP[i]+FN[i])
        FPR[i] = FP[i]/(FP[i]+TN[i])
        PPV[i] = TP[i]/(TP[i]+FP[i])
        F1[i] = 2* PPV[i]*TPR[i]/(PPV[i]+TPR[i])

    #printing output
    print('TP: ',TP)
    print('TN: ',TN)
    print('FP: ',FP)
    print('FN: ',FN)
    print('Acc: ',sum(Acc)/size, Acc)
    print('recall: ',sum(TPR)/size,TPR)
    print('FPR: ',sum(FPR)/size,FPR)
    print('precision: ',sum(PPV)/size,PPV)
    print('F1: ',sum(F1)/size,F1)

    #producing heatmap form Cmatrix
    fig, ax = plt.subplots()
    ax.matshow(Cmatrix, cmap=plt.cm.Reds)
    for i in range(size):
        for j in range(size):
            c = Cmatrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(name)

class Classifier:
    """
    base classifier class with parallel processing
    n_features: number of features
    n_jobs: number of parallel cores to use in computation
    Xt: training dataset features
    Yt: training dataset labels
    """

    n_features = 2
    # Intiation method
    # n_jobs number of parallel processes to run using joblib
    def __init__(self, n_jobs=1):
        self.jobs = n_jobs
        if n_jobs == -1:
            self.jobs = joblib.cpu_count()
        self.n_features = 2

    # Training method
    def fit(self, Xt, Yt):
        self.Xt = Xt
        self.Yt = Yt
        self.n_features = len(Xt[0])

    # Classification predictor
    def predict(self, Xq):
        return None

    # Classification predictor (sequential processing)
    def predict_seq(self, Xq):
        return None

    # Plot model using parallel processing (good for performance testing)
    def plot(self, xmin1, xmax1, xmin2, xmax2, n_grid=20, x1=0, x2=1):
        xp, yp, cp = [], [], []
        Xp = [0 for i in range(self.n_features)]
        for i in range(n_grid + 1):
            x = i / n_grid * (xmax1 - xmin1) + xmin1
            for j in range(n_grid + 1):
                y = j / n_grid * (xmax2 - xmin2) + xmin2
                Xp[x1] = x
                Xp[x2] = y
                c = self.predict(Xp)
                xp.append(x)
                yp.append(y)
                cp.append(c)

        labels = np.asarray(cp)
        unique = np.unique(labels)
        color = []
        for label in labels:
            color.append(np.where(unique == label)[0][0])

        plt.scatter(xp, yp, c=color)
        plt.show()

    # Plot model using sequential processing
    def plot_seq(self,  xmin1, xmax1, xmin2, xmax2,n_grid=20, x1=0, x2=0):
        xp, yp, cp = [], [], []
        Xp = [0 for i in range(self.n_features)]
        for i in range(n_grid + 1):
            x = i / n_grid * (xmax1 - xmin1) + xmin1
            for j in range(n_grid + 1):
                y = j / n_grid * (xmax2 - xmin2) + xmin2
                Xp[x1] = x
                Xp[x2] = y
                c = self.predict_seq(Xp)
                xp.append(x)
                yp.append(y)
                cp.append(c)

        labels = np.asarray(cp)
        unique = np.unique(labels)
        color = []
        for label in labels:
            color.append(np.where(unique == label)[0][0])

        plt.scatter(xp, yp, c=color)
        plt.show()


# Decision tree classifier class
class DecisionTreeClassifier(Classifier):
    """
    Decision Tree Classifier
    root: root node
    log_output: flag to switch printing tree output on/off
    tree_depth: Maximum tree depth
    split_metric: Gini or Entropy
    """
    root = None
    log_output = False

    def __init__(self, tree_depth=400, split_metric='gini', min_gain=1e-7):
        self.tree_depth = tree_depth
        if split_metric == 'gini':
            self.split_metric = gini_impurity
        else:
            self.split_metric = entropy
        self.min_gain = min_gain

    class node:
        """
        Tree node class
        value: decision value
        feature: decision feature
        left_branch: left branch node
        right_branch: right branch node
        """

        def __init__(self, value, feature, left_branch, right_branch):
            self.value = value
            self.feature = feature
            self.left_branch = left_branch
            self.right_branch = right_branch

    class leaf:
        """
        Tree leaf class
        label: decision label
        """

        def __init__(self, label):
            self.label = label

    def split_search(self, rows, counts, sum_counts, Xt, Xt_fast, Yt_indexed):
        """
        Searching for best decision split
        :param rows: list of rows in current node
        :param counts: list of label counts
        :param sum_counts: total number of counts
        :param Xt: features training set
        :param Xt_fast: same as Xt but in a list (faster performance)
        :param Yt_indexed: indecies of labels
        :return: right and left split results, and info gain from best split
        """
        # Intitating variables, assume all labels are going right (True)
        max_gain = -1e10
        best_split_value = None
        best_split_feature = None
        left_counts = [0] * len(counts)
        best_sum_left = 0
        right_counts = list(counts)
        best_sum_right = sum_counts
        best_left_counts = list(left_counts)
        best_right_counts = list(right_counts)
        best_left_rows_list = [False for i in range(len(rows))]
        Xst = [Xt_fast[row] for row in rows]
        best_sorted_rows = list(rows)
        best_left_rows = []
        best_right_rows = []

        #timers for checking speed performance
        t1 = 0
        t2 = 0
        t3 = 0

        # Iterate over n_features
        for column in range(Xt.shape[1]):
            t0 = time.time()

            # Extract current feature column
            column_Xt = lambda x, i: [x[j][i] for j in range(len(x))]
            Xtt = column_Xt(Xst, column)

            # Sort rows according to the current feature value to speed performance
            sorted_rows = [x for _, x in sorted(zip(Xtt, rows))]
            t1 += time.time() - t0
            t0 = time.time()

            # Intiations
            last_split_index = 0
            left_counts = [0] * len(counts)
            sum_left = 0
            right_counts = list(counts)
            sum_right = sum_counts
            last_split_value = None
            left_rows = [False for row in rows]
            t2 += time.time() - t0

            # Iterate over sorted rows
            for i, row in enumerate(sorted_rows):

                # Choose split value
                split_value = Xt_fast[row][column]

                # Make sure split value is different from last value (to save time)
                if last_split_value is None or split_value > last_split_value:
                    last_split_value = split_value

                    # Iterate from last split index to current split (i) only to update
                    # left and right label counts.
                    for j in range(last_split_index, i):
                        left_rows[j] = True
                        index = Yt_indexed[sorted_rows[j]]
                        left_counts[index] += 1
                        sum_left += 1
                        right_counts[index] -= 1
                        sum_right -= 1

                    last_split_index = i

                    # Calculate information gain from current split
                    # Some speed can be gained if we can compute incremental info gain (not implemented)
                    t0 = time.time()
                    # Use fast version of info gain (this line below is the bottle neck in performance)
                    gain = info_gain2(left_counts,
                                      sum_left,
                                      right_counts,
                                      sum_right,
                                      self.split_metric)
                    t3 += time.time() - t0

                    # Check if we have better information gain and save bunch of variables
                    if gain > max_gain:
                        max_gain = gain
                        best_split_value = split_value
                        best_split_feature = column
                        best_left_counts = list(left_counts)
                        best_sum_left = sum_left
                        best_right_counts = list(right_counts)
                        best_sum_right = sum_right
                        best_left_rows_list = list(left_rows)
                        best_sorted_rows = list(sorted_rows)

        # Construct best left and right rows lists from best split
        for i in range(len(sorted_rows)):
            if best_left_rows_list[i]:
                best_left_rows.append(best_sorted_rows[i])
            else:
                best_right_rows.append(best_sorted_rows[i])

        # Best split is found, returning bunch of variables
        best_info_gain = info_gain(counts,
                                   best_left_counts,
                                   best_right_counts,
                                   self.split_metric)

        if self.log_output and best_split_value is not None:
            print('Sorting time =', t1, '\nt1=', t1, '\nt2=', t2, '\nt3=', t3)
            print('Split value =', best_split_value,
                  '\nSplit feature =', best_split_feature,
                  '\ninfo gain =', best_info_gain,
                  '\nYes counts =', best_right_counts,
                  '\nNO counts =', best_left_counts)

        # Split found, now return results
        return best_left_counts, best_sum_left, best_right_counts, best_sum_right, \
               best_info_gain, best_split_value, best_split_feature, best_left_rows, best_right_rows

    def grow_tree(self,
                  rows,
                  labels,
                  counts,
                  sum_counts,
                  depth,
                  Xt,
                  Xt_fast,
                  Yt_indexed):
        """
        Growing Decision tree recrusively
        """

        t0 = time.time()

        left_counts, \
        sum_left, \
        right_counts, \
        sum_right, \
        gain, value, \
        feature, \
        left_rows, \
        right_rows = self.split_search(rows,
                                       counts,
                                       sum_counts,
                                       Xt,
                                       Xt_fast,
                                       Yt_indexed)

        if self.log_output:
            print("Split Search time =", time.time() - t0)

        # Test to see if it's a leaf
        if gain < self.min_gain or (self.tree_depth != -1 and depth >= self.tree_depth):
            if self.log_output:
                print('-' * depth, 'Leaf node, gain =', gain)
            return DecisionTreeClassifier.leaf(
                labels[max(enumerate(counts),
                           key=lambda y: y[1])[0]])

        depth += 1

        # First we recurresively build the right branch
        if self.log_output:
            print()
            print('-' * depth, 'Yes branch')
        right_branch = self.grow_tree(right_rows,
                                      labels,
                                      right_counts,
                                      sum_right,
                                      depth,
                                      Xt,
                                      Xt_fast,
                                      Yt_indexed)

        # Then we go for the left branch
        if self.log_output:
            print()
            print('-' * depth, 'No branch')
        left_branch = self.grow_tree(left_rows,
                                     labels,
                                     left_counts,
                                     sum_left,
                                     depth,
                                     Xt,
                                     Xt_fast,
                                     Yt_indexed)

        # Tree is grown
        return DecisionTreeClassifier.node(value, feature, left_branch, right_branch)

    def fit(self, Xt, Yt, output=False):
        """
        Train a decision tree model
        Where Xt: is training data
              Yt: is training labels
        """


        self.n_features = len(Xt[0])
        labels, counts = np.unique(Yt, return_counts=True)

        # Output training output
        self.log_output = output
        if self.log_output:
            print('Root node')

        # Feature selection

        # create dictionary of labels
        labels_dict = {}
        Yt_indexed = []
        for i in range(len(labels)):
            labels_dict[labels[i]] = i

        # Index all labels in training set according to their frequency
        # To gain speed
        for i in range(len(Yt)):
            if isinstance(Yt[i], str):
                Yt_indexed.append(labels_dict[str(np.squeeze(Yt[i]))])
            else:
                Yt_indexed.append(labels_dict[int(np.squeeze(Yt[i]))])

        # normal list of lists is faster than numpy array
        Xt_fast = Xt.tolist()

        # Train decision tree
        # Convert labels & counts to lists to speed operations
        self.root = self.grow_tree(
            [i for i in range(len(Yt))],
            labels.tolist(),
            counts.tolist(),
            sum(counts.tolist()),
            0,
            Xt,
            Xt_fast,
            Yt_indexed)

    def predict(self, Xq):
        """
        Evaluate data against trained model
        """
        this_node = self.root
        leaf_found = False

        # loop unitl we find a leaf
        while not leaf_found:
            if type(this_node) is DecisionTreeClassifier.node:
                # Evaluate which decision path to take
                if Xq[this_node.feature] >= this_node.value:
                    this_node = this_node.right_branch
                else:
                    this_node = this_node.left_branch
            else:
                # Leaf found, exit loop
                leaf_found = True

        # Return prediction label (leaf label)
        return this_node.label

    def save(self, file = ''):
        """
        Saves a decision tree to .json file
        :param file: file name
        :return:
        """

        def return_node_dict(node):
            """
            Create node dictionary to output as .json file
            :param node: Node class instance
            :return: node dictionary
            """

            node_dict = {}
            if type(node) is DecisionTreeClassifier.leaf:
                node_dict['leaf'] = True
                node_dict['label'] = node.label
            else:
                node_dict['leaf'] = False
                node_dict['feature'] = node.feature
                node_dict['value'] = node.value
                node_dict['right_branch'] = return_node_dict(node.right_branch)
                node_dict['left_branch'] = return_node_dict(node.left_branch)
            return node_dict

        Tree_dict = return_node_dict(self.root)
        Tree_dict['n_features'] = self.n_features
        if file != '':
            try:
                with open(file, 'w') as fp:
                    json.dump(Tree_dict,fp)
            except:
                print('Error saving json file!')
        return Tree_dict

    def load(self, file):
        """
        Load decision tree from .json file
        :param file: file name
        :return:
        """
        def return_node(node_dict):
            if node_dict['leaf']:
                node = DecisionTreeClassifier.leaf(node_dict['label'])
            else:
                node = DecisionTreeClassifier.node(node_dict['value'],
                                                   node_dict['feature'],
                                                   return_node(node_dict['left_branch']),
                                                   return_node(node_dict['right_branch']))
            return node
        try:
            with open(file, 'r') as fp:
                node_dict = json.load(fp)
                self.n_features = node_dict['n_features']
                self.root = return_node(node_dict)
        except:
            print('Error loading json file')


class RandomForestClassifier(Classifier):
    """
    Random Forest Classifier class
    """
    forest = []

    class TreeClass:
        """
        Tree class for each decision tree in random forest
        """
        # out of bag index
        OOB_index = []
        DecisionTree = None

        def __init__(self,
                     tree_depth,
                     split_metric,
                     output,
                     lenX):
            self.DecisionTree = DecisionTreeClassifier(tree_depth=tree_depth,
                                                       split_metric=split_metric)
            self.tree_depth = tree_depth
            self.split_metric = split_metric
            self.features_index = []
            self.OOB_index = []

            # number of features = sqrt(total number of features)
            n_features = int(math.sqrt(lenX))

            # Randomly sample features without replacement
            self.features_index = np.sort(np.random.choice(lenX,
                                                           size=n_features,
                                                           replace=False))

        def grow(self, Xt, Yt, output):
            """
            Grow a decision tree for random forest
            :param Xt: Training set features
            :param Yt: Training set labels
            :param output:
            :return:
            """

            Xtree = np.zeros((Xt.shape[0], len(self.features_index)))

            # Create a set from selected features
            for i, j in np.ndenumerate(self.features_index):
                Xtree[:, i[0]] = Xt[:, j]

            self.DecisionTree.fit(Xtree, Yt, output)

        def predict(self, Xq):
            # Map features to selected features in this tree and return a prediction
            Xq_tree = []
            for index in self.features_index:
                Xq_tree.append(Xq[index])
            return self.DecisionTree.predict(Xq_tree)

    def __init__(self,
                 n_trees=10,
                 tree_depth=400,
                 split_metric='gini',
                 n_jobs=1,
                 random_state=0):
        Classifier.__init__(self, n_jobs=n_jobs)
        self.forest = []
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.split_metric = split_metric
        np.random.RandomState(random_state)

    def grow_a_tree(self, tree):

        Xtrain = np.asarray(self.Xt)
        Xbagging = np.zeros((len(self.Xt), len(self.Xt[0])))
        Ybagging = []

        # bagging with replacement
        X_index = np.random.choice(len(self.Xt),
                                   size=Xbagging.shape[0],
                                   replace=True)

        # collecting Out-of-bag samples indicies
        tree.OOB_index = np.squeeze(np.setdiff1d(np.arange(len(X_index)),
                                                 X_index)).tolist()

        # Constructing bagged features set and labels
        for i, j in np.ndenumerate(X_index):
            Xbagging[i[0], :] = Xtrain[j, :]
            Ybagging.append(self.Yt[j])

        tree.grow(Xbagging, Ybagging, False)
        return tree

    def fit(self, Xt, Yt, output=False):
        """
        Training a random forest
        :param Xt:
        :param Yt:
        :param output:
        :return:
        """
        Classifier.fit(self, Xt, Yt)
        for i in range(self.n_trees):
            self.forest.append(self.TreeClass(self.tree_depth,
                                              self.split_metric,
                                              output,
                                              len(Xt[0])))

        # growing trees in parallel to speed up training
        self.forest = Parallel(n_jobs=self.jobs, verbose=10) \
            (delayed(self.grow_a_tree)(tree) for tree in self.forest)

    def get_oob(self, trees):
        """
        Out-of-bag accuracy
        :param trees: Number of trees to consider
        :return: oob
        """
        results = 0
        for i, xt in enumerate(self.Xt):
            result = []

            for tree in range(trees):
                if i in self.forest[tree].OOB_index:
                    result.append(self.forest[tree].predict(xt))

            unique, counts = np.unique(result, return_counts=True)

            if len(counts) > 0 and unique[counts.argmax()] == self.Yt[i]:
                results += 1


        return results / len(self.Xt)

    def plot_oobe(self, name, start=15, end=-1, step=1):
        """
        Out-of-bag error plot
        :param name: image file name
        :param start: start tree number
        :param end: end tree number
        :param step: step
        :return:
        """

        if end == -1:
            end = self.n_trees

        error = Parallel(n_jobs=self.jobs, verbose=20)(delayed(self.get_oob)(tree) for tree in range(start, end, step))
        error = [1-error[i] for i in range(len(error))]
        plt.plot([i for i in range(start, end, step)], error)
        plt.xlabel('No. of trees')
        plt.ylabel('Out-of-bag error')
        plt.title('Out-of-bag error plot')
        plt.savefig(name)

    def predict(self, Xq, n_trees):
        """
        Predict a label from data using trained classifier
        :param Xq:
        :param n_trees:
        :return:
        """
        results = []
        for i in range(n_trees):
            results.append(self.forest[i].predict(Xq))
        unique, counts = np.unique(results, return_counts=True)
        return unique[counts.argmax()]

    def save(self, file):
        """
        Save trained Random Forest classifier to .json file
        :param file:
        :return:
        """
        forest_dict = {}
        forest_dict['n_trees'] = self.n_trees
        forest_dict['n_features'] = self.n_features
        forest_dict['Xt'] = np.asarray(self.Xt).tolist()
        forest_dict['Yt'] = np.asarray(self.Yt).tolist()
        forest_OOB_list = []
        forest_trees = []
        forest_features_index = []
        for tree in self.forest:
            forest_OOB_list.append(tree.OOB_index)
            forest_features_index.append(tree.features_index.tolist())
            forest_trees.append(tree.DecisionTree.save())
        forest_dict['OOB_list'] = forest_OOB_list
        forest_dict['features_index'] = forest_features_index
        forest_dict['trees'] = forest_trees

        try:
            with open(file, 'w') as fp:
                json.dump(forest_dict, fp)
        except:
            print('Error saving json file')

    def load(self, file):
        """
        load trained random forest classifier from .json
        :param file:
        :return:
        """

        def return_node(node_dict):
            if node_dict['leaf']:
                node = DecisionTreeClassifier.leaf(node_dict['label'])
            else:
                node = DecisionTreeClassifier.node(node_dict['value'],
                                                   node_dict['feature'],
                                                   return_node(node_dict['left_branch']),
                                                   return_node(node_dict['right_branch']))
            return node

        try:
            with open(file, 'r') as fp:
                forest_dict = json.load(fp)
                self.n_trees = forest_dict['n_trees']
                self.n_features = forest_dict['n_features']
                self.Xt = np.asarray(forest_dict['Xt'])
                self.Yt = forest_dict['Yt']
                forest_OOB_list = forest_dict['OOB_list']
                forest_features_index = forest_dict['features_index']
                forest_trees = forest_dict['trees']
                for OOB, index, tree in zip(forest_OOB_list, forest_features_index, forest_trees):
                    tree_object = RandomForestClassifier.TreeClass(self.tree_depth,
                                                                   self.split_metric,
                                                                   False,
                                                                   len(self.Xt[0]))
                    tree_object.OOB_index = OOB
                    tree_object.features_index = index
                    tree_object.DecisionTree.n_features = tree['n_features']
                    tree_object.DecisionTree.root = return_node(tree)
                    # tree_object.DecisionTree.load(tree)
                    self.forest.append(tree_object)
        except:
            print('Error reading json file')


