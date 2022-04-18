#!/usr/bin/env python3
#

import sys
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import joblib
import tensorflow as tf
import tensorflow.keras as keras

################################################################
#
# Data/File functions
#
def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def get_data(filename):
    data = pd.read_csv(filename)
    return data

def load_data(my_args, filename):
    data = get_data(filename)
    data = data.sample(frac=1.0) # sample 100% of the data (causes shuffling to occur)
    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    y = data[label_column]
    return X, y

def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file
#
# Data/File functions
#
################################################################


################################################################
#
# Pipeline classes and functions
#
class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at the current stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.mCategoricalPredictors = []
        self.mNumericalPredictors = [ "{}".format(i) for i in range(1,785) ]
        self.mLabels = ["labels"]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def getCategoricalPredictors(self):
        return self.mCategoricalPredictors

    def getNumericalPredictors(self):
        return self.mNumericalPredictors

    def fit( self, X, y=None ):
        # no fit necessary
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values

def make_numerical_feature_pipeline(my_args):
    items = []
    
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))

    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Numerical Preprocessing")))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    ###
    ### sklearn's decision tree classifier requires all input features to be numerical
    ### one hot encoding accomplishes this.
    ###
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Categorial Preprocessing")))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []
    dfs = DataFrameSelector()
    if len(dfs.getNumericalPredictors()) > 0:
        items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    if len(dfs.getCategoricalPredictors()) > 0:
        items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def make_pseudo_fit_pipeline(my_args):
    """
    Pipeline that can be used for prepreocessing of data, but
    the model is blank because the model is a Tensorflow network.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Final Preprocessing")))
    items.append(("model", None))
    return sklearn.pipeline.Pipeline(items)
#
# Pipeline functions
#
################################################################


################################################################
#
# CNN functions
#

def create_model(my_args, input_shape):
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(filters=10, kernel_size=(7,7), strides=(1,1), activation="relu", padding="valid",
                                  input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(1,1), activation="relu", padding="valid"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(25, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    # model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", padding="same",
                                  # input_shape=input_shape))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(256, activation="relu"))
    # model.add(keras.layers.Dense(128, activation="relu"))
    # model.add(keras.layers.Dense(64, activation="relu"))
    # model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], optimizer=keras.optimizers.Adam())
    print(model.summary())

    return model

def do_cnn_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_pseudo_fit_pipeline(my_args)
    pipeline.fit(X)
    X = pipeline.transform(X) # If the resulting array is sparse, use .todense()
    # reshape the 784 pixels into a 2D greyscale image
    X = np.reshape(X,[X.shape[0],28,28,1])
    
    model = create_model(my_args, X.shape[1:])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, verbose=1, callbacks=[early_stopping], validation_split=my_args.validation_split)

    # save the last file
    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump((pipeline, model), model_file)
    return
#
# CNN functions
#
################################################################

################################################################
#
# Evaluate existing models functions
#
def sklearn_metric(y, yhat):
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    ###
    header = "+"
    for col in range(cm.shape[1]):
        header += "-----+"
    rows = [header]
    for row in range(cm.shape[0]):
        row_str = "|"
        for col in range(cm.shape[1]):
            row_str += "{:4d} |".format(cm[row][col])
        rows.append(row_str)
    footer = header
    rows.append(footer)
    table = "\n".join(rows)
    print(table)
    print()
    ###
    if cm.shape[0] == 2:
        precision = sklearn.metrics.precision_score(y, yhat)
        recall = sklearn.metrics.recall_score(y, yhat)
        f1 = sklearn.metrics.f1_score(y, yhat)
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
    else:
        report = sklearn.metrics.classification_report(y, yhat)
        print(report)
    return

def show_score(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    if my_args.show_test:
        test_file = get_test_filename(my_args.test_file, train_file)
        if not os.path.exists(test_file):
            raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    basename = get_basename(train_file)

    X_train, y_train = load_data(my_args, train_file)
    if my_args.show_test:
        X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    if isinstance(pipeline, tuple):
        (pipeline, model) = pipeline
        X_train = pipeline.transform(X_train) # .todense()
        # reshape the 784 pixels into a 2D greyscale image
        X_train = np.reshape(X_train,[X_train.shape[0],28,28,1])
        yhat_train = np.argmax(model.predict(X_train), axis=1)
        print()
        print("{}: train: ".format(basename))
        print()
        sklearn_metric(y_train, yhat_train)
        print()

        if my_args.show_test:
            X_test = pipeline.transform(X_test) # .todense()
            X_test = np.reshape(X_test,[X_test.shape[0],28,28,1])
            yhat_test = np.argmax(model.predict(X_test), axis=1)
            print()
            print("{}: test: ".format(basename))
            print()
            print()
            sklearn_metric(y_test, yhat_test)
            print()

    else:
        yhat_train = pipeline.predict(X_train)
        print()
        print("{}: train: ".format(basename))
        print()
        sklearn_metric(y_train, yhat_train)
        print()

        if my_args.show_test:
            yhat_test = pipeline.predict(X_test)
            print()
            print("{}: test: ".format(basename))
            print()
            print()
            sklearn_metric(y_test, yhat_test)
            print()
        
    return
#
# Evaluate existing models functions
#
################################################################



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image Classification with CNN')
    parser.add_argument('action', default='cnn-fit',
                        choices=[ "cnn-fit", "score" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")

    #
    # Pipeline configuration
    #
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="labels",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str,   help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--print-preprocessed-data', default=0,         type=int,   help="0 = don't do the debugging print, 1 = do print (default=0)")

    #
    # hyper parameters
    #
    parser.add_argument('--validation-split', default=0.1,         type=float,   help="validation split fraction (default=0.1)")

    # debugging/observations
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")


    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args


def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'cnn-fit':
        do_cnn_fit(my_args)
    elif my_args.action == 'score':
        show_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    
