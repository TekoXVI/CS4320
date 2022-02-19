#!/usr/bin/env python3

import sys
import pandas as pd
import sklearn.metrics

def get_data(filename):
    """
    ### Assumes column 0 is the instance index stored in the
    ### csv file.  If no such column exists, remove the
    ### index_col=0 parameter.

    Assumes the column named "Cabin" should be a interpreted 
    as a string, but Pandas can't figure that out on its own.

    ###Request missing values (blank cells) to be left as empty strings.

    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    """

    ###, index_col=0
    ###, keep_default_na=False
    data = pd.read_csv(filename, dtype={ "Cabin": str })
    return data


def predict_women_survive(data):
    result = data["Sex"] == "female"
    return result

def predict_children_survive(data, age):
    result = data["Age"] < age
    return result

def predict_women_children_survive(data, age):
    result = (data["Sex"] == "female") | (data["Age"] < age)
    return result

def tree_predict_survive(data, age):
    """
                     female?
                    /       \
                  yes       no
                  |          |
             Pclass?         age < 10
            /   |   \       /        \
           1    2    3     yes        no
           |    |    |     |           |
     survive survive die  survive     die

    predict women survive if first or second class, but not if third class
    predict men survive is less than age, but not if equal to or more than age

    """
    result1 = (data["Sex"] == "female") & ((data["Pclass"] == 1) | (data["Pclass"] == 2))
    result2 = (data["Sex"] != "female") & (data["Age"] < age)
    result = result1 | result2
    return result


def manual_metric(y, yhat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(yhat)):
        if yhat[i]:
            if y.iloc[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y.iloc[i] == 1:
                FN += 1
            else:
                TN += 1

    precision = TP/(TP+FP)            
    recall = TP/(TP+FN)
    f1 = 2.0 / ( (1.0/precision) + (1.0/recall) )
    table = "+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n".format(TN, FN, FP, TP)
    print(table)
    print()
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))
    return

def sklearn_metric(y, yhat):
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    table = "+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n".format(cm[0][0], cm[1][0], cm[0][1], cm[1][1])
    print(table)
    print()
    precision = sklearn.metrics.precision_score(y, yhat)
    recall = sklearn.metrics.recall_score(y, yhat)
    f1 = sklearn.metrics.f1_score(y, yhat)
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))
    return


def main(argv):
    filename = "titanic-train.csv"
    data = get_data(filename)

    predictions = {}
    predictions["female"] = predict_women_survive(data)
    predictions["children"] = predict_children_survive(data, 10)
    predictions["female or children"] = predict_women_children_survive(data, 10)
    predictions["tree"] = tree_predict_survive(data, 10)

    for key in predictions:
        print()
        print("========================================")
        print()
        print("Prediction Criteria: {}".format(key))
        sklearn_metric(data["Survived"], predictions[key])
        print()

    # for age in range(1,100):
    #     f1 = sklearn.metrics.f1_score(data["Survived"], predictions)
    #     print(age, f1)
    #
    # manual_metric(data["Survived"], predictions)
    # print()
    # print()
    #
    # sklearn_metric(data["Survived"], predictions)

    return

if __name__ == "__main__":
    main(sys.argv)
