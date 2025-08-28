"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *


np.random.seed(42)


class Node:
    def __init__(self, feature_index = None, threshold = None, left=None, right = None, info_gain = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def build_Tree(self,dataset:pd.DataFrame,  current_depth = 0):
        X  = dataset.iloc[:, :-1]
        y = dataset['y']
        if(current_depth<=self.max_depth):  
            best_split = opt_split_attribute(X,y, self.criterion)
            # print(best_split)
            if self.criterion != "mse":
                if not best_split or best_split.get("info_gain", 0) <= 0:
                    Y = list(y)
                    leaf = max(Y, key=Y.count)
                    return Node(value=leaf)
            else:
                if not best_split or best_split.get("mse", 0) <= 0:
                    leaf = y.mean()
                    return Node(value=leaf)
            if self.criterion!="mse":
                if best_split["info_gain"]>0:
                    left_child = self.build_Tree(best_split["data_left"], current_depth+1)
                    right_child = self.build_Tree(best_split["data_right"], current_depth+1)
                    return Node(best_split['index'], best_split['threshold'], left_child, right_child, best_split['info_gain'] )
            else:
                if best_split["mse"]>0:
                    left_child = self.build_Tree(best_split["data_left"], current_depth+1)
                    right_child = self.build_Tree(best_split["data_right"], current_depth+1)
                    return Node(best_split['index'], best_split['threshold'], left_child, right_child, best_split['mse'] )
        if self.criterion != "mse":    
            Y = list(y)
            leaf = max(Y, key=Y.count)
            return Node(value=leaf)
        else:
            leaf = y.mean()
            return Node(value=leaf)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        if(check_ifreal(X.iloc[:,0])==False):
            X = one_hot_encoding(X)
        #case of discrete output
        if(check_ifreal(y) == False):
            X['y'] = y
            self.root = self.build_Tree(X, 0)
            self.print_tree()
        else:
            X['y'] = y
            self.root = self.build_Tree(X,0)

       

        pass

    def prediction_traversal(self, X, node):
        if node.value!=None: 
            return node.value #reached leaf node
        curr_value = X[node.feature_index]
        if curr_value<=node.threshold:
            return self.prediction_traversal(X, node.left)
        else:
            return self.prediction_traversal(X, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        predictions = [self.prediction_traversal(rows, self.root) for _, rows in X.iterrows()]
        return predictions
        

    def plot(self, node,x,y, dx, dy, ax ):
        ax.text(x, y, str(node.value), ha="center", va="center",
            bbox=dict(facecolor="skyblue", boxstyle="circle", edgecolor="black"))
        
        if node.left:
            ax.plot([x, x - dx], [y - 0.1, y - dy + 0.1], color="black")
            self.plot(node.left, x - dx, y - dy, dx/2, dy, ax)
        
        if node.right:
            ax.plot([x, x - dx], [y - 0.1, y - dy + 0.1], color="black")
            self.plot(node.left, x - dx, y - dy, dx/2, dy, ax)
    

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        pass
        # if not tree:
        #     tree = self.root

        # if tree.value is not None:
        #     print(tree.value)

        # else:
        #     print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
        #     print("%sleft:" % (indent), end="")
        #     self.print_tree(tree.left, indent + indent)
        #     print("%sright:" % (indent), end="")
        #     self.print_tree(tree.right, indent + indent)
