# MOHAN RAO DIVATE KODANDARAMA
# divatekodand@wisc.edu
# CS USERID: divate-kodanda-rama

# For Python 2 / 3 compatability
from __future__ import print_function
import sys


from scipy.io import arff
from io import StringIO
import numpy as np
import math
#import matplotlib.pyplot as plt

class Node:
    """ Node holds the best split, childrens of the input data according to the best split

          The compare method compares the feature value of the example with the feature value
          of the best split.
    """

    def __init__(self, feature_no, value, childrens, features_used, num_pos, num_neg):
        self.feature_no = feature_no
        self.value = value
        # self.feature_type = feature_type
        self.childrens = childrens
        self.features_used = features_used
        self.num_pos = num_pos
        self.num_neg = num_neg

    def compare(self, example):
        # The compare method compares the feature value of the example with the feature value
        # of the best split.
        global train_attribute_types
        val = example[self.feature_no]
        if train_attribute_types[self.feature_no] == 'numeric':
            return val <= self.value
        else:
            i = 0
            for value in train_attribute_ranges[self.feature_no][1]:
                if (value == val):
                    break;
                else:
                    i += 1;
            return i  # Returns the index of the nominal value


class Leaf:
    """A Leaf node contains the class label
    """

    def __init__(self, label, features_used, num_pos, num_neg):
        self.label = label
        self.features_used = features_used
        self.num_pos = num_pos
        self.num_neg = num_neg

class Split:
    """Split consists of a feature and a threshold for a numeric feature. For nominal Feature only feature is stored"""

    def __init__(self, feature_no, feature_type, threshold):
        self.feature_no = feature_no
        self.feature_type = feature_type
        self.threshold = threshold

#CODE FOR Q2 AND Q3
# def find_accuracy(tree, test_set):
#     num_correct = 0
#     num_incorrect = 0
#     predictions = predict(test_set, tree)
#     for i in range(test_set.size):
#         #print(str(i + 1) + ": Actual: " + test_set[i][-1].decode('UTF-8') + " Predicted: " + str(predictions[i]))
#         if (test_set[i][-1].decode('UTF-8') == predictions[i]):
#             num_correct += 1
#         else:
#             num_incorrect += 1
#     # print("Number of correctly classified: " + str(num_correct) + " Total number of test instances: " + str(test_set.size))  #,end="")
#     t_accuracy = (num_correct / (test_set.size * 1))
#     return t_accuracy
#
# def hw2_q2(dataset, testset):
#     '''
#     Generates plot for hw2 q2
#     :param dataset:
#     :return:
#     '''
#     # print(type(dataset))
#     # print(type(dataset[1]))
#     # print(dataset[1])
#     n_dataset = len(dataset)
#     global split_treshold, tree_str
#     num_samples = 10
#     data_sizes = [5,10,20,50,100]
#     acc = []
#     for val in data_sizes:
#         n_train = int(round(n_dataset * (val / 100.0)))
#         # print(n_train)
#         accuracy = []
#         for i in range(num_samples):
#             if i > 0 and val == data_sizes[-1]:
#                 accuracy.append(accuracy_sample)
#                 continue
#             p_dataset = np.random.choice(dataset, size=n_train, replace=False)
#             # print(p_dataset)
#             # print(type(p_dataset))
#             f_used = []
#             num_pos = 0
#             for example in p_dataset:
#                 # print(type(example))
#                 # print(example)
#                 if example[-1].decode('UTF-8') == 'positive':
#                     num_pos += 1
#             if num_pos >= (n_train - num_pos):
#                 label = 'positive'
#             else:
#                 label = 'negative'
#             tree = build_tree(p_dataset, f_used, label)
#             # tree_str = ""
#             # print_tree(tree)
#             # print(tree_str)
#             accuracy_sample = find_accuracy(tree, testset)
#             accuracy.append(accuracy_sample)
#         acc.append(accuracy)
#     assert(len(acc) == len(data_sizes))
#     np_acc = np.array(acc)
#     np_acc = np_acc.T
#     print(np_acc)
#     mins = np_acc.min(0)
#     maxes = np_acc.max(0)
#     means = np_acc.mean(0)
#     std = np_acc.std(0)
#     # plt.errorbar(data_sizes, means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1, capsize=10)
#     # plt.xlim(0, 105)
#     # plt.ylim(0, 1)
#     # plt.show()
#     # input("Press Enter to continue...")


def is_stop_criteria_met(dataset, features_used, best_gain):
    """ Checks if the stopping criteria is met
    """
    global split_treshold
    global num_features
    global only_nominal

    #if (dataset is None or features_used is None or best_gain is None):
        #print("Null input to is_stop_criteria_met")
    outputs = []
    for data in dataset:
        outputs.append(data[-1])
    # print(outputs)
    num_outputs = len(set(outputs))
    if (num_outputs < 2):  ## All examples belong to the same class
        return True
    elif (len(
            dataset) < split_treshold):  ##No of examples reaching this node is less than the threshold specified in the cmd
        return True
    ##TODO: check the information gain criteria
    elif (best_gain < 0):  ##No feature has positive infomation gain
        return True
    elif (len(features_used) == num_features and only_nominal):
        return True
    else:
        return False


def find_entropy(dataset):
    """Finds the entropy of the ouput RV for the given dataset
    """
    #print("find entropy")
    # print(dataset)
    total_examples = len(dataset)

    #if (total_examples == 0):
        #print("E/W - Number of training examples to the find_entropy tree function is 0")
    # print("dataset" , dataset)
    num_pos = 0
    num_neg = 0
    for data in dataset:
        if (data[-1] == b'negative'):
            num_neg += 1
        elif (data[-1] == b'positive'):
            num_pos += 1
        else:
            None #print("E/W: class of an example is neither positive nor negative", data[-1])
    p = (num_pos / (total_examples * 1.0))
    #print("find entropy p = " + str(p))
    if p == 0. or p == 1.:
        return 0
    else:
        entropy = - (p * math.log(p, 2) + (1 - p) * math.log((1 - p), 2))
    return entropy


def partition(dataset, split):
    global train_attribute_ranges
    data_partitions = []
    if (split.feature_type == 'numeric'):
        left_data = []
        right_data = []
        dataset_size = len(dataset)
        for i in range(dataset_size):
            if dataset[i][split.feature_no] <= split.threshold:
                left_data.append(dataset[i])
            else:
                right_data.append(dataset[i])
        data_partitions.append(left_data)
        data_partitions.append(right_data)
    else:
        values = train_attribute_ranges[split.feature_no][1]
        #print(type(values), type(train_attribute_ranges[split.feature_no][1]))
        n_values = len(values)
        for value in values:
            one_partition = []
            for data in dataset:
                #print(type(data[split.feature_no]), type(value))
                #print(data[split.feature_no].decode('UTF-8'), value)
                if data[split.feature_no].decode('UTF-8') == value:
                    one_partition.append(data)
            data_partitions.append(one_partition)

    return data_partitions


def compute_info_gain(data_partitions, current_entropy):
    info_gain = current_entropy
    n_examples = 0
    pos_count = 0
    neg_count = 0
    n_partitions = len(data_partitions)
    for partition in data_partitions:
        n_examples += len(partition)

    #if n_examples == 0:
        #print("n_examples = ", n_examples)

    for partition in data_partitions:
        n_partition = len(partition)
        if n_partition != 0.:
            partition_entropy = find_entropy(partition)
        else:
            partition_entropy = 0
        info_gain -= (n_partition / (n_examples * 1.0)) * partition_entropy

    return info_gain


def find_best_split(dataset, features_used):
    """Find the best split by iterating over every feature / value
    and calculating the information gain."""

    global num_features
    global train_attributes
    global train_attribute_types
    #print("find best split")
    max_gain = 0
    best_feature_no = 0
    best_feature_type = None
    best_feature_threshold = None
    current_entropy = find_entropy(dataset)
    #print("find entropy succeded")

    for feature_no in range(num_features):
        unique_vals = set([data[feature_no] for data in dataset])
        unique_vals = list(unique_vals)  # Elements in sets cannot be sorted. Hence convert to list
        if (train_attribute_types[feature_no] == 'numeric'):
            unique_vals.sort()
            n_vals = len(unique_vals)
            for i in range(n_vals - 1):
                mid = (unique_vals[i] + unique_vals[i + 1]) / 2.0
                split = Split(feature_no, 'numeric', mid)
                data_partitions = partition(dataset, split)
                if len(data_partitions[0]) == 0 or len(data_partitions[1]) == 0:
                    continue

                info_gain = compute_info_gain(data_partitions, current_entropy)

                if (info_gain > max_gain):
                    max_gain = info_gain
                    best_feature_no = feature_no
                    best_feature_type = 'numeric'
                    best_feature_threshold = mid
        else:
            if len(features_used) != 0 and train_attributes[feature_no] in features_used:
                continue
            split = Split(feature_no, 'nominal', 'Nominal')
            data_partitions = partition(dataset, split)
            # TODO: Should I handle the case where one of the partition is null?
            info_gain = compute_info_gain(data_partitions, current_entropy)
            if (info_gain > max_gain):
                max_gain = info_gain
                best_feature_no = feature_no
                best_feature_type = 'nominal'
                best_feature_threshold = 'Nominal'

    best_split = Split(best_feature_no, best_feature_type, best_feature_threshold)

    return max_gain, best_split


def build_tree(data, features_used, parent_label):
    """
      Builds the tree and returns the root node.
    """
    total_examples = len(data)
    ## Number of Training examples = 0. This case should never arrive as this is handled in the while creating nodes
    if (total_examples == 0):
        #print("E/W - Number of training examples to the build tree function is 0")
        return Leaf(parent_label, features_used, 0, 0)
    else:
        #print("find best split call - ", data, features_used)
        best_gain, best_split = find_best_split(data, features_used)
        #print("find best split succeded")
        num_pos = 0
        for example in data:
            #print(type(example))
            #print(example)
            if example[-1].decode('UTF-8') == 'positive':
                num_pos += 1
        if num_pos > (total_examples - num_pos):
            label = 'positive'
        elif num_pos < (total_examples - num_pos):
            label = 'negative'
        else:
            label = parent_label
        if (is_stop_criteria_met(data, features_used, best_gain)):
            leaf = Leaf(label, features_used, num_pos, (total_examples - num_pos))
            return leaf
        else:
            features_used.append(best_split.feature_no)
            data_partitions = partition(data, best_split)
            nodes = []
            for data_p in data_partitions:
                node = build_tree(data_p, features_used, label)
                nodes.append(node)
            current_node = Node(best_split.feature_no, best_split.threshold, nodes, features_used, num_pos, (total_examples - num_pos))
            return current_node


def print_tree(node, spacing=""):
    """print the tree"""
    global train_attributes
    global train_attribute_ranges
    global tree_str
    # Base case
    if isinstance(node, Leaf):
        #print(": ", node.label)
        #tree_str = tree_str + ": " + node.label  # + "\n"
        return

    # Print the question at this node
    #print(spacing + str(node.feature_no) + str(node.value))

    # Call this function recursively on the true branch
    i = 0
    #print(train_attribute_ranges)
    for child in node.childrens:
        if(train_attribute_ranges[node.feature_no][0] == 'numeric'):
            if(i == 0):
                #print(spacing + train_attributes[node.feature_no] +  " <= " + str(node.value)),
                dist = "[" + str(child.num_neg) + " " + str(child.num_pos) + "]"
                tree_str = tree_str + "\n" + spacing + train_attributes[node.feature_no] +  " <= " + "%.6f" % node.value + " " + dist
                #tree_str = tree_str + "" + spacing + train_attributes[node.feature_no] + " <= " + str(node.value) + "\n"
                if isinstance(node.childrens[0], Leaf):
                    tree_str = tree_str + ": " + node.childrens[0].label
            else:
                #print(spacing + train_attributes[node.feature_no] + " > " + str(node.value)),
                dist = "[" + str(child.num_neg) + " " + str(child.num_pos) + "]"
                tree_str = tree_str + "\n" + spacing + train_attributes[node.feature_no] + " > " + "%.6f" % node.value + " " + dist
                #tree_str = tree_str + "" + spacing + train_attributes[node.feature_no] + " > " + str(node.value) + "\n"
                if isinstance(node.childrens[1], Leaf):
                    tree_str = tree_str + ": " + node.childrens[1].label
        else:
            #print(spacing + train_attributes[node.feature_no] +  " = " + train_attribute_ranges[node.feature_no][-1][i]),
            dist = "[" + str(child.num_neg) + " " + str(child.num_pos) + "]"
            tree_str  = tree_str + "\n" + spacing + train_attributes[node.feature_no] +  " = " + train_attribute_ranges[node.feature_no][-1][i] + " " + dist
            #tree_str = tree_str + "" + spacing + train_attributes[node.feature_no] + " = " + train_attribute_ranges[node.feature_no][-1][i] + "\n"
            if isinstance(node.childrens[i], Leaf):
                tree_str = tree_str + ": " + node.childrens[i].label
        i += 1
        print_tree(child, spacing + "|\t"),


def predict(dataset, tree):
    global train_attribute_types
    global train_attribute_ranges
    global train_attributes
    labels = []
    for data in dataset:
        node = tree
        while not isinstance(node, Leaf):
            if train_attribute_types[node.feature_no] == 'nominal':
                i = 0
                for val in train_attribute_ranges[node.feature_no][-1]:
                    #print(data[node.feature_no])
                    #print(type(data[node.feature_no]))
                    #print(val)
                    #print(type(val))
                    #print(data[node.feature_no].decode('UTF-8'))
                    if data[node.feature_no].decode('UTF-8') == val:
                        node = node.childrens[i]
                        # if isinstance(node, Leaf):
                        #     print(node.label)
                        # else:
                        #     print(train_attributes[node.feature_no])
                        break;
                    i += 1
            else:
                #print(data[node.feature_no])
                #print(type(data[node.feature_no]))
                #print(node.value)
                #print(type(node.value))
                if data[node.feature_no] <= node.value:
                    node = node.childrens[0]
                    # if isinstance(node, Leaf):
                    #     print(node.label)
                    # else:
                    #     print(train_attributes[node.feature_no])
                else:
                    node = node.childrens[1]
                    # if isinstance(node, Leaf):
                    #     print(node.label)
                    # else:
                    #     print(train_attributes[node.feature_no])
        labels.append(node.label)
    return labels


def main():
    '''
    Loads the data
    '''
    ##PROCESS THE ARGURMENT
    # print sys.argv
    num_args = len(sys.argv)
    if (num_args < 4):
        print("Wrong Usage - Script takes 3 arguments")
        print("Example Usage- python dt-learn.py heart_train.arff heart_test.arff 2")
        exit(0)
    train_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    global split_treshold
    split_treshold = int(sys.argv[3])
    # print train_set_filename, test_set_filename, split_treshold

    ##LOAD THE DATA
    global train_attributes
    global train_attribute_types
    global num_examples
    global num_features
    global train_attribute_ranges

    train_set_file = open(train_set_filename, 'r')
    train_set, train_meta = arff.loadarff(train_set_file)
    num_examples = len(train_set)
    num_features = len(train_set[1]) - 1
    # print(train_set)
    # print train_set.size
    train_attributes = train_meta.names()  # Returns a list
    train_attribute_types = train_meta.types()  # Returns a list
    #print(train_attribute_types)

    # To find if there are only nominal features - this is one of the stopping criteria when there are only nominal fetures
    global only_nominal
    only_nominal = False
    set_of_train_attr_types = set(train_attribute_types)
    # print(set_of_train_attr_types)
    if (2 != len(set_of_train_attr_types)):
        only_nominal = True
        #print("Nominal features only")

    train_attribute_ranges = []
    num_attributes= len(train_attributes)

    for i in range(num_attributes):
        #print(type(train_meta.__getitem__(train_attributes[i])))
        train_attribute_ranges.append(train_meta.__getitem__(train_attributes[i]))

    #print(train_attribute_ranges)
    #print('negative' == train_attribute_ranges[-1][-1][0])
    # print train_attributes
    # print train_attribute_types
    test_set_file = open(test_set_filename, 'r')
    test_set, test_meta = arff.loadarff(test_set_file)
    test_attributes = test_meta.names()  # Returns a list
    test_attribute_types = test_meta.types()  # Returns a list
    test_attribute_ranges = []

    for i in range(num_attributes):
        test_attribute_ranges.append(test_meta.__getitem__(test_attributes[i]))

    #print(test_attributes, test_attribute_types)

    ## BUILD THE TREE AND PRINT THE RESULT ON THE TEST SET
    total_examples = len(train_set)
    num_pos = 0
    for example in train_set:
        if example[-1] == 'positive':
            num_pos += 1
    if num_pos >= (total_examples - num_pos):
        label = 'positive'
    else:
        label = 'negative'
    features_used = []
    #hw2_q2(train_set, test_set)
    tree = build_tree(train_set, features_used, label)
    global tree_str
    tree_str = ""
    print_tree(tree)
    tree_str = tree_str[1:]
    print(tree_str)

    num_correct = 0
    num_incorrect = 0
    predictions = predict(test_set, tree)
    print("<Predictions for the Test Set Instances>")
    #print(test_set.size)
    # predictions = range(test_set.size)
    for i in range(test_set.size):
        print(str(i + 1) + ": Actual: " + test_set[i][-1].decode('UTF-8') + " Predicted: " + str(predictions[i]))
        if (test_set[i][-1].decode('UTF-8') == predictions[i]):
            num_correct += 1
        else:
            num_incorrect += 1
    print("Number of correctly classified: " + str(num_correct) + " Total number of test instances: " + str(test_set.size))  #,end="")


if __name__ == "__main__":
    main()
