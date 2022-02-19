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
    data = pd.read_csv(filename, dtype={ "age": int })
    return data

#age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall 

def predict_sex(data):
    result = data["sex"] == 0
    return result

def predict_age(data, age):
    result = data["age"] > age
    return result

def predict_chest_pain(data):
    # Value 1: typical angina
    # Value 2: atypical angina
    # Value 3: non-anginal pain
    # Value 4: asymptomatic
    result = data["cp"] == 4
    return result
    
def predict_resting_blood_pressure(data, value):
    # in mm Hg
    result = data["trtbps"] > value
    return result
    
def predict_cholesterol(data, value):
    # mg/dl
    result = data["chol"] > value
    return result

def predict_fasting_blood_sugar(data):
    # > 120 mg/dl = 1 (true) or 0 (false)
    result = data["fbs"] == 0
    return result

def predict_resting_ecg_results(data, value):
    # Value 0: normal
    # Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    # Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    result = data["restecg"] == value
    return result

def predict_max_heart_rate(data, value):
    # maximum heart rate achieved
    result = data["thalachh"] < value
    return result

def predict_exercise_induced_angina(data):
    # exercuse induced angina t/f 1/0
    result = data["exng"] == 1
    return result

def predict_number_of_major_vessels(data, value):
    # number of major blood vessels (0-3)
    result = data["caa"] < value
    return result



'''def tree_predict_survive(data, age):
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
'''

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
    filename = "heart-train.csv"
    data = get_data(filename)

    predictions = {}
    predictions["female"] = predict_sex(data)
    predictions["age"] = predict_age(data, 70)
    predictions["chest pain"] = predict_chest_pain(data)
    predictions["resting blood pressure"] = predict_resting_blood_pressure(data, 125)
    predictions["cholesterol"] = predict_cholesterol(data, 215)
    predictions["fasting blood sugar > 120 mg/dl"] = predict_fasting_blood_sugar(data)
    predictions["resting ecg results"] = predict_resting_ecg_results(data, 2)
    predictions["max heart rate"] = predict_max_heart_rate(data, 150)
    predictions["exercise induced angina"] = predict_exercise_induced_angina(data)
    predictions["number of major blood vessels"] = predict_number_of_major_vessels(data, 2)

    for key in predictions:
        print()
        print("========================================")
        print()
        print("Prediction Criteria: {}".format(key))
        sklearn_metric(data["output"], predictions[key])
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
