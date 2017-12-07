import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import collections
from collections import Counter
from scipy import stats as st
from sklearn.model_selection import KFold


class Node():
    level = 1
    split_attribute = 0
    split_test = 0
    leaf = False
    label = 0
    lf_node = None
    rt_node = None

    # arrType = arrythmia type
    def __init__(self, level=1, split_attribute=None, split_test=None, leaf=False, label=None):
        self.level = level
        self.split_attribute = split_attribute
        self.split_test = split_test
        self.leaf = leaf
        self.label = label
        return

    def __str__(self):
        if not self.leaf:
            string = "Internal Node, Attribute: {}, Test value (<=) {}, Level: {}".format(self.split_attribute,
                                                                                          self.split_test, self.level)
        else:
            string = "Leaf Node, Label: {}, Level: {}".format(self.label, self.level)

        return (string)


class DecisionTree():
    max_depth = 10000
    root_node = 0


    def test():
        print("does this work")
        return

    def tp():
        test()
        return

    def entropy(data):
        y = data[headers[-1]]
        val, freqs = np.unique(y, return_counts=True)
        probs = freqs.astype("float") / len(y)
        return -probs.dot(np.log2(probs))

    def entropy_from_dict(count_dict, row_len):
        num_lt, num_gt, ent_lt, ent_gt = 0, 0, 0, 0

        for label in count_dict.keys():
            freq_lt = count_dict[label]["<"]
            freq_gt = count_dict[label][">"]

            if freq_lt != 0.0:
                ent_lt -= (freq_lt / row_len) * np.log2(freq_lt / row_len)
                num_lt += freq_lt
            if freq_gt != 0.0:
                ent_gt -= (freq_gt / row_len) * np.log2(freq_gt / row_len)
                num_gt += freq_gt
        return (num_lt / row_len) * ent_lt + (num_gt / row_len) * ent_gt

    # iterate over test_value to find best test value, then find best header is index,
    def info_gain(data, header, test_value):
        headers = data.dtypes.index

        d_lf = data[data[header] <= test_value]
        d_rt = data[data[header] > test_value]

        # calc entropy for data perhaps this is being calc twice
        e_root = entropy(data)
        e_lf = entropy(d_lf)
        e_rt = entropy(d_lf)

        n_root = len(data)
        n_lf = len(d_lf)
        n_rt = len(d_lf)

        e_avg = (n_lf / n_root) * e_lf + (n_lf / n_root) * e_rt

        return e_root - e_avg

    # recursively build tree
    def recursiveTree(data, headers, current_depth=1, max_depth=2, root_label=None):
        # print("recursiveTree depth: {}, max: {}, label: {}".format(current_depth,max_depth, root_label))

        y = data[headers[-1]]

        # Stopping condition for depth and size of data and purity of y
        if stop(y, current_depth, max_depth):
            labelType = classify(y, root_label)
            return Node(level=current_depth, leaf=True, label=labelType)

        # calculate best attribute, create root node
        best_attribute, best_test_value = best_attribute_cont(data, headers)
        root = Node(level=current_depth, split_attribute=best_attribute, split_test=best_test_value)

        # increment current depth by 1, calc root_label,
        new_depth = current_depth + 1
        new_root_label = classify(y, root_label)

        # split data based on best attribute and value
        lf_data = data[data[best_attribute] <= best_test_value]
        # print("lf_data length {}".format(len(lf_data[headers[-1]])))
        rt_data = data[data[best_attribute] > best_test_value]
        # print("rf_data length {}".format(len(rt_data[headers[-1]])))

        # remove tested column
        lf_data = lf_data.drop(best_attribute, 1)
        rt_data = rt_data.drop(best_attribute, 1)

        new_headers = lf_data.dtypes.index

        # create new left tree and right tree for those
        lf_tree = recursiveTree(data=lf_data, headers=new_headers, current_depth=new_depth, max_depth=max_depth,
                                root_label=new_root_label)
        rt_tree = recursiveTree(data=rt_data, headers=new_headers, current_depth=new_depth, max_depth=max_depth,
                                root_label=new_root_label)

        root.lf_node = lf_tree
        root.rt_node = rt_tree

        return root

    # for cont functions
    def best_attribute_cont(data, headers):
        # find best attribute and test to split on based on continuous entropy test for each column
        # find lowest entropy for each col with continuous split

        lowest_entropy = entropy(data)  # max entropy will be the entropy of unsplit values
        best_attribute = 0
        best_test_value = 0
        best_row = 0
        col_len = len(data.columns)
        row_len = 0

        # for each attribute
        for i in range(0, col_len - 1):
            header = headers[i]
            attribute_data = data[[header, headers[-1]]]
            sorted_attribute_data = attribute_data.sort_values((header))
            count_labels = attribute_data[headers[-1]].value_counts()
            row_len = len(attribute_data[headers[-1]])

            # dict to track frequency of lt and gt
            count_dict = {label: {"<": 0, ">": count} for label, count in count_labels.iteritems()}

            # for each row
            for j in range(0, row_len):
                cur_label = sorted_attribute_data.iloc[j, 1]

                # for every label encountered, increment frequency of less than, and decrement freq of gt
                count_dict[cur_label]["<"] = count_dict[cur_label][">"] + 1
                count_dict[cur_label][">"] = count_dict[cur_label][">"] - 1

                # cal entropy with after update freq.
                cur_entropy = entropy_from_dict(count_dict, row_len)
                # print("Cur_Ent {} ".format(cur_entropy))

                # if lower entropy set best attribute to header
                if cur_entropy < lowest_entropy:
                    best_attribute = header
                    lowest_entropy = cur_entropy
                    best_row = j
                    best_test_value = sorted_attribute_data.iloc[best_row, 0]
                    # print("Best attribute {}, best row {}".format(best_attribute, best_row))

        print("Best attribute {}, best value {}".format(best_attribute, best_test_value))
        return best_attribute, best_test_value

    # get root mode label
    def stop(y, current_depth, max_depth):
        toStop = False
        if len(set(y)) in [0, 1] or current_depth >= max_depth:
            return True
        return toStop

    # recurse on root model label
    def classify(y, root_label):
        if y.empty:
            return root_label
        return st.mode(y)[0][0]

    def print_BFS(root):
        if root is None:
            return
        queue = [root]
        while queue:
            vertex = queue.pop(0)
            if vertex.lf_node:
                queue.append(vertex.lf_node)
            if vertex.rt_node:
                queue.append(vertex.rt_node)
            print(vertex)
        return

    # embed in predict w/ self.root node
    # self.root node
    def generate_predictions(Xdata, tree_root):
        # check every row to determine tree prediction
        # return array of labels

        # stopping conditions: no data,
        if Xdata.size == 0:
            return

        # may not be necessary
        # prevent error when last row is autoconverted to 1D array
        if (Xdata.shape) == 1:
            Xdata = np.reshape(Xdata, (1, len(Xdata)))

        # predictions empty
        predictions = np.zeros(len(Xdata))

        for i in range(0, len(Xdata)):
            node = tree_root
            # print(node.label)
            while not node.leaf:
                data_value = Xdata.loc[i, node.split_attribute]

                if data_value <= node.split_test:
                    node = node.lf_node
                else:
                    node = node.rt_node
            predictions[i] = node.label
            # print(predictions[i])
        return predictions

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        return

    def fit(self, X, y):

        dataS = pd.concat([X, y], axis=1)
        dataS.columns = ["X" + str(i) for i in range(1, len(dataS.columns) + 1)]
        headers = dataS.dtypes.index

        self.root_node = recursiveTree(data=dataS, headers=headers, current_depth=1, max_depth=self.max_depth,
                                       root_label=None)

        return

    def predict(self, T):

        T.columns = ["X" + str(i) for i in range(1, len(T.columns) + 1)]
        headers = T.dtypes.index

        return generate_predictions(T, self.root_node)

    def print(self):
        # use breadth_first_search
        print_BFS(self.root_node)
        return


def validation_curve():
    # read file arrhythmia.csv

    # get sample data
    data = pd.DataFrame.from_csv("arrhythmia.data", header=None, index_col=False)
    y = data.iloc[:, 279]
    X = data.iloc[:, :279]

    # get subset of X and y
    Xs = X.iloc[:30, :10]
    ys = y.iloc[:30]

    # randomly divide the data into three partitions
    # T1 = a + b, D1 = c
    # T2 = b + c, D2 = a
    # T3 = a + c, D3 = b
    A, B, C = np.split(Xs.sample(frac=1), [int((1.0 / 3.0) * len(Xs)), int((2.0 / 3.0) * len(Xs))])
    T1 = pd.concat([A, B])
    D1 = C
    T2 = pd.concat([B, C])
    D2 = A
    T3 = pd.concat([A, C])
    D3 = B

    # create a decision tree and fit using the first two partitions
    # predict (calculate accuracy of tree) using first two partitions
    # predict using D1
    # repeat for sets t2,t3 and d2, d3,
    # calculate average accuracy for training sets,
    # calculate average accuracy for test sets
    # repeat for max depth from 2 to 20 in steps of 2

    err_training = np.zeros(10)
    err_test = np.zeros(10)

    for max_depth in range(2, 22, 2):
        ind_err_training = np.zeros(3)
        ind_err_test = np.zeros(3)
        for data in [(T1, D1), (T2, D2), (T2, T3)]:
            tree = DecisionTree(max_depth=max_depth)
            tree.fit()

    # plot average training set accuracy and test set accuracy for 11 steps


    return

def calc_error(predictions, y):
    #check predictions for equality with y
    #increment error count if inequality
    #return error fraction
    err = 0
    total = len(predictions)

    for i in range(0,total):
        if predictions[i] != y[0][i]:
            err = err + 1
    return err/total


tree = DecisionTree(max_depth = 3)
tree.tp()


#retry format data

df = pd.DataFrame.from_csv("arrhythmia.data", header=None, index_col = False)
y = df.iloc[:,279]
X = df.iloc[:,:279]

for c in X.columns:
    most_common = Counter([float(x) for x in X.iloc[:,c] if x!='?']).most_common(1)[0][0]  # Finds most frequent element
    for r in range(X.shape[0]):
        cur = X.iloc[r,c]
        if cur=='?':
            X.iloc[r,c]=most_common
        elif type(cur) is str:
            X.iloc[r,c]=int(float(cur))


_X = X
_y = y

X = _X.iloc[:30,:10]
y = _y.iloc[:30]

err_training = np.zeros(10)
err_test = np.zeros(10)

# Cross validation: need to be passed as np.array
Xs = np.array(X)
ys = np.array(y)

kf = KFold(n_splits=2)
kf.get_n_splits(Xs)

print(kf)

for train_index, test_index in kf.split(Xs):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xs[train_index], Xs[test_index]
    y_train, y_test = ys[train_index], ys[test_index]

    for max_depth in range(2, 22, 2):
        # convert to df
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)

        # fit tree to train and determine error
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)
        pred = tree.predict(X_train)
        err_training[int(max_depth / 2 - 1)] += calc_error(pred, y_train)

        # determine error for test set
        pred_test = tree.predict(X_test)
        err_training[int(max_depth / 2 - 1)] += calc_error(pred_test, y_test)
    print(err_training)
    print(err_test)

# average errors
err_training /= 3
err_test /= 3

print("Training Error: {}".format(err_training))
print("Test Error: {}".format(err_test))




