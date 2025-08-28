from typing import Union
import pandas as pd
from tree.utils import *
import math

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    n = len(y)
    classified_correct = 0
    assert len(y_hat) == len(y)
    for ele1, ele2 in zip(y, y_hat):
        if ele1 == ele2:
            classified_correct = classified_correct+1
    
    return ((classified_correct)/n)
    # TODO: Write here


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    total_positives = 0
    false_positives = 0
    assert len(y_hat) == len(y)
    for ele1, ele2 in zip(y, y_hat):
        if((ele1 == cls) and (ele2==cls)):
            total_positives = total_positives + 1
    for ele1, ele2 in zip(y, y_hat):
        if((ele1 ==cls) and (ele2!=cls)):
            false_positives = false_positives+1 
    
    return total_positives/(total_positives+false_positives)
    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    total_positives = 0
    false_negatives = 0
    assert len(y_hat) == len(y)
    for ele1, ele2 in zip(y, y_hat):
        if((ele1 == cls) and (ele2==cls)):
            total_positives = total_positives + 1
    for ele1, ele2 in zip(y, y_hat):
        if((ele1!=cls) and (ele2==cls)):
            false_negatives = false_negatives+1 
    
    return total_positives/(total_positives+false_negatives)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    squared_diff = (y_hat - y) ** 2
    mse = squared_diff.mean()
    rmse = math.sqrt(mse)
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return (y_hat - y).abs().mean()
