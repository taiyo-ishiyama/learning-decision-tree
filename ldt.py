import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node: # Node to store learning decision tree
  def __init__(self, attribute=None, children=None, class_value=None):
    self.attribute = attribute # selected attribute
    self.children = children # children node
    self.class_value = class_value # value in class if leaf node

def split_dataset(file_name): # split dataset and balance training and testing data
  data = pd.read_csv(file_name)
  data = data.sample(frac=1, random_state=106).reset_index(drop=True) # shuffle data
  attributes = list(data.columns[:-1]) # list of attributes

  train_ratio = 0.8 # 80% (training) : 20% (testing)

  classes = data["class"].unique()
  train_list = []
  test_list = []

  for value_class in classes: # split dataset in each evaluation (eg. "good", "vgood")
    data_in_class = data[data["class"] == value_class]
    index = int(len(data_in_class) * train_ratio)
    train_list.append(data_in_class.iloc[:index])
    test_list.append(data_in_class.iloc[index:])

  # concatenate all examples
  train_data = pd.concat(train_list).sample(frac=1).reset_index(drop=True)
  test_data = pd.concat(test_list).sample(frac=1).reset_index(drop=True)
  return train_data, test_data, attributes

def learn_decision_tree(examples, attributes, parent_examples, max_depth=10, min_examples=5, depth=0): # building learning decision tree recursively
  if len(examples) == 0: # if example is empty
    return Node(class_value=plurality_value(parent_examples))
  elif len(examples["class"].unique()) == 1: # if one label is remaining in class
    return Node(class_value=examples["class"].unique()[0])
  elif len(attributes) == 0 or depth >= max_depth or len(examples) < min_examples: # pruning tree by defining depth and minimum examples
    return Node(class_value=plurality_value(examples))
  else:
    best_attribute = select_best_attribute(examples, attributes) # choose next attribute
    new_node = Node(attribute=best_attribute, children={})
    values_in_attributes = examples[best_attribute].unique()
    for value in values_in_attributes: # add branch for each value in attribute
        exs = examples[examples[best_attribute] == value]
        sub_attributes = attributes.copy()
        sub_attributes.remove(best_attribute)
        subtree = learn_decision_tree(exs, sub_attributes, examples, max_depth, min_examples, depth + 1) # recursive
        new_node.children[value] = subtree
    return new_node

def plurality_value(examples): # get plurality value
  value_dict = examples["class"].value_counts().to_dict()
  max_value = 0
  max_class = None
  for value in examples["class"].unique(): # find the value with largest number of examples
    if value_dict.get(value) == None:
      continue
    if value_dict[value] > max_value:
      max_class = value
  return max_class

def calculate_entropy(values): # entropy for information gain
  value, num_values = np.unique(values, return_counts=True)
  ratio = num_values / len(values)
  return -np.sum(ratio * np.log2(ratio))

def calculate_ig(parent_entropy, child_values): # information gain for selecting attribute
  total_num_values = 0.0
  for values in child_values:
    total_num_values += len(values)
  child_entropy = 0.0
  for values in child_values:
    child_entropy += (len(values) / total_num_values) * calculate_entropy(values)

  return parent_entropy - child_entropy

def select_best_attribute(examples, attributes): # select the attribute with maximum information gain
    best_attribute = None
    max_ig = 0.0
    parent_entropy = calculate_entropy(examples["class"])

    for attribute in attributes: # iterate each remaining attribute
        child_values = [examples[examples[attribute] == value]["class"] for value in examples[attribute].unique()] # get data for each label in attribute
        ig = calculate_ig(parent_entropy, child_values)
        if ig > max_ig:
            max_ig = ig
            best_attribute = attribute
    return best_attribute

def predict(node, test_data): # predict output for a given input data
  if node.class_value is not None: # if leaf node is found
    return node.class_value
  else:
    next_node = node.children.get(test_data[node.attribute]) # get corresponding node
    if next_node is not None:
      return predict(next_node, test_data) # recursive
    else:
      return None

def test_LDT(node, test_dataset): # go through every input data to predict
  predicted_values = []
  for _, input in test_dataset.iterrows():
    predicted_value = predict(node, input)
    predicted_values.append(predicted_value) # store predicted outputs
  return predicted_values

def calculate_accuracy(actual_values, predicted_values): # get accuracy for predicted outputs
   num_correct = np.sum(np.array(actual_values) == np.array(predicted_values))
   return num_correct / len(actual_values)

def calculate_confusion_matrix(actual_values, predicted_values): # get confusion matrix
    num_classes = len(np.unique(actual_values))
    classes = np.unique(actual_values)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # initialise matrix
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for true_label, pred_label in zip(actual_values, predicted_values): # count each output
        confusion_matrix[class_to_index[true_label], class_to_index[pred_label]] += 1

    return confusion_matrix

def components_for_metrics(actual_values_binary, predicted_values_binary):
    # get true positive, true negative, false positive, false negative
    tp = sum(1 for i in range(len(actual_values_binary)) if actual_values_binary[i] == 1 and predicted_values_binary[i] == 1)
    tn = sum(1 for i in range(len(actual_values_binary)) if actual_values_binary[i] == 0 and predicted_values_binary[i] == 0)
    fp = sum(1 for i in range(len(actual_values_binary)) if actual_values_binary[i] == 0 and predicted_values_binary[i] == 1)
    fn = sum(1 for i in range(len(actual_values_binary)) if actual_values_binary[i] == 1 and predicted_values_binary[i] == 0)
    return tp, tn, fp, fn

def calculate_metrics(actual_values, predicted_values): # get all required metrics
  num_classes = len(np.unique(actual_values))
  num_data = len(actual_values)
  list_classes = np.unique(actual_values)

  macro_precision, macro_recall, macro_f1 = 0, 0, 0
  weighted_precision, weighted_recall, weighted_f1 = 0, 0, 0
  macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

  confusion_matrix = calculate_confusion_matrix(actual_values, predicted_values)

  for class_value in list_classes: # compute metrics for each label
    actual_values_binary = [1 if i == class_value else 0 for i in actual_values]
    predicted_values_binary = [1 if i == class_value else 0 for i in predicted_values]
    tp, tn, fp, fn = components_for_metrics(actual_values_binary, predicted_values_binary)

    support = sum(actual_values_binary) # support value for weighted average

    # precision
    partial_precision = tp / (tp + fp + 1e-6)
    macro_precision += partial_precision
    weighted_precision += partial_precision * (support / num_data)

    macro_precision_list.append(partial_precision)

    # recall
    partial_recall = tp / (tp + fn + 1e-6)
    macro_recall += partial_recall
    weighted_recall += partial_recall * (support / num_data)

    macro_recall_list.append(partial_recall)

    # F1
    partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall + 1e-6)
    macro_f1 += partial_f1
    weighted_f1 += partial_f1 * (support / num_data)

    macro_f1_list.append(partial_f1)

  macro_precision /= num_classes
  macro_recall /= num_classes
  macro_f1 /= num_classes

  return confusion_matrix, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, macro_precision_list, macro_recall_list, macro_f1_list


def print_tree(node, depth=0): # print the decision tree
  if node is None:
    return
  if node.class_value is not None:
    print(' ' * depth, 'Predicted evaluation:', node.class_value)
  else:
    print(' ' * depth, 'Attribute:', node.attribute)
    for value, subtree in node.children.items():
      print(' ' * depth, 'Value:', value)
      print_tree(subtree, depth + 5)

def plot_learning_curve(train_data, test_data, attributes): # plot learning curve accuracy vs percentage of learning examples
  train_sizes = np.linspace(0.05, 1.0, 20) # every 5% of data input
  list_accuracies = []

  for train_size in train_sizes:
    subset_data = train_data.sample(frac=train_size, random_state=5)
    tree_node = learn_decision_tree(subset_data, attributes, None, max_depth=10, min_examples=5)
    predicted_values = test_LDT(tree_node, test_data)
    accuracy = calculate_accuracy(test_data['class'], predicted_values)
    list_accuracies.append(accuracy) # store every accuracy

  plt.plot(train_sizes * 100, list_accuracies, marker='o', label='Learning Curve')
  plt.xlabel('Percentage of Learning Examples')
  plt.ylabel('Accuracy')
  plt.title('Learning Curve')
  plt.legend()
  plt.grid()
  plt.show()


file_name = "car.csv"
train_data, test_data, attributes = split_dataset(file_name)

tree_node = learn_decision_tree(train_data, attributes, None, max_depth=10, min_examples=5, depth=0)

predicted_values = test_LDT(tree_node, test_data)
actual_values = list(test_data['class'])

print("\nSize of the training set:", len(train_data))
print("Size of the testing set:", len(test_data))

total_accuracy = calculate_accuracy(actual_values, predicted_values)
print("\nTotal accuracy:", total_accuracy)

confusion_matrix, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, macro_precision_list, macro_recall_list, macro_f1_list = calculate_metrics(actual_values, predicted_values)

print("\nConfusion matrix:\n", confusion_matrix)
print("")

for index, class_value in enumerate(train_data['class'].unique()):
    print(class_value + ": \nPrecision with macro average: " + str(macro_precision_list[index]) + "\nRecall with macro average: " + str(macro_recall_list[index]) + "\nF1 with macro average: " + str(macro_f1_list[index]) + "\n")

print("\nTotal precision with macro-average: ", macro_precision)
print("Total recall with macro-average: ", macro_recall)
print("Total F1 with macro-average: ", macro_f1)
print("Total precision with weighted-average: ", weighted_precision)
print("Total recall with weighted-average: ", weighted_recall)
print("Total F1 with weighted-average: ", weighted_f1)

print("")

print_tree(tree_node)

plot_learning_curve(train_data, test_data, attributes)
