import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import consts
from enum import Enum


class Classifier(Enum):
    LASSO = 1
    RANDOM_FOREST = 2


def main():
    # Load the data
    non_overlapping_two_class_data = get_data(consts.DATA_PATH + r'\10_merged_block_feature_space(Big Data).csv')
    overlapping_two_class_data = get_data(consts.DATA_PATH + r'\10_random_merged_block_feature_space.csv')
    non_overlapping_sliding_two_class_data = get_data(
        consts.DATA_PATH + r'\10_merged_sliding_block_feature_space.csv')
    overlapping_sliding_two_class_data = get_data(
        consts.DATA_PATH + r'\10_random_merged_sliding_block_feature_space.csv')
    non_overlapping_three_class_data = get_data(consts.DATA_PATH + r'\102_merged_block_feature_space.csv')
    overlapping_three_class_data = get_data(consts.DATA_PATH + r'\102_random_merged_block_feature_space.csv')
    non_overlapping_sliding_three_class_data = get_data(
        consts.DATA_PATH + r'\102_merged_sliding_block_feature_space.csv')
    overlapping_sliding_three_class_data = get_data(
        consts.DATA_PATH + r'\102_random_merged_sliding_block_feature_space.csv')

    # Split the data into training and test sets.
    non_overlapping_block_train, non_overlapping_block_test = split_data(non_overlapping_two_class_data)
    overlapping_block_train, overlapping_block_test = split_data(overlapping_two_class_data)
    non_overlapping_sliding_block_train, non_overlapping_sliding_block_test = split_data(
        non_overlapping_sliding_two_class_data)
    overlapping_sliding_block_train, overlapping_sliding_block_test = split_data(overlapping_sliding_two_class_data)
    non_overlapping_three_class_train, non_overlapping_three_class_test = split_data(non_overlapping_three_class_data)
    overlapping_three_class_train, overlapping_three_class_test = split_data(overlapping_three_class_data)
    non_overlapping_sliding_three_class_train, non_overlapping_sliding_three_class_test = split_data(
        non_overlapping_sliding_three_class_data)
    overlapping_sliding_three_class_train, overlapping_sliding_three_class_test = split_data(
        overlapping_sliding_three_class_data)

    # Store the split data
    store_split_data(non_overlapping_block_train, r'\10_non_overlapping_block_train.csv', non_overlapping_block_test,
                     r'\10_non_overlapping_block_test.csv')
    store_split_data(overlapping_block_train, r'\10_overlapping_block_train.csv', overlapping_block_test,
                     r'\10_overlapping_block_test.csv')
    store_split_data(non_overlapping_sliding_block_train, r'\10_non_overlapping_sliding_block_train.csv',
                     non_overlapping_sliding_block_test, r'\10_non_overlapping_sliding_block_test.csv')
    store_split_data(overlapping_sliding_block_train, r'\10_overlapping_sliding_block_train.csv',
                     overlapping_sliding_block_test, r'\10_overlapping_sliding_block_test.csv')
    store_split_data(non_overlapping_three_class_train, r'\102_non_overlapping_three_class_train.csv',
                     non_overlapping_three_class_test, r'\102_non_overlapping_three_class_test.csv')
    store_split_data(overlapping_three_class_train, r'\102_overlapping_three_class_train.csv',
                     overlapping_three_class_test, r'\102_overlapping_three_class_test.csv')
    store_split_data(non_overlapping_sliding_three_class_train, r'\102_non_overlapping_sliding_three_class_train.csv',
                     non_overlapping_sliding_three_class_test, r'\102_non_overlapping_sliding_three_class_test.csv')
    store_split_data(overlapping_sliding_three_class_train, r'\102_overlapping_sliding_three_class_train.csv',
                     overlapping_sliding_three_class_test, r'\102_overlapping_sliding_three_class_test.csv')

    # Show histograms of the data.
    show_histogram(non_overlapping_block_train, non_overlapping_block_test)
    show_histogram(non_overlapping_sliding_block_train, non_overlapping_sliding_block_test)
    show_histogram(non_overlapping_three_class_train, non_overlapping_three_class_test)
    show_histogram(non_overlapping_sliding_three_class_train, non_overlapping_sliding_three_class_test)
    show_histogram(overlapping_block_train, overlapping_block_test)
    show_histogram(overlapping_sliding_block_train, overlapping_sliding_block_test)
    show_histogram(overlapping_three_class_train, overlapping_three_class_test)
    show_histogram(overlapping_sliding_three_class_train, overlapping_sliding_three_class_test)

    # Train the Lasso models for two class datasets.
    train_with_lasso(non_overlapping_two_class_data)
    train_with_lasso(overlapping_two_class_data)
    train_with_lasso(non_overlapping_sliding_two_class_data)
    train_with_lasso(overlapping_sliding_two_class_data)

    # Train the Random Forest models.
    train_with_random_forest(non_overlapping_two_class_data)
    train_with_random_forest(overlapping_two_class_data)
    train_with_random_forest(non_overlapping_sliding_two_class_data)
    train_with_random_forest(overlapping_sliding_two_class_data)
    train_with_random_forest(non_overlapping_three_class_data)
    train_with_random_forest(overlapping_three_class_data)
    train_with_random_forest(non_overlapping_sliding_three_class_data)
    train_with_random_forest(overlapping_sliding_three_class_data)


def get_data(path):
    """
    Loads the data from the given path.
    :param path: The path to the data.
    """
    return pd.read_csv(path, skiprows=1, header=None, index_col=None)


def split_data(data):
    """
    Splits the data into training and test sets.
    :param data: The data to split.
    """
    train_data, test_data = train_test_split(data, test_size=0.22, shuffle=False)
    print("Training set: \n{}".format(train_data))
    print("Test set: \n{}".format(test_data))
    return train_data, test_data


def store_split_data(train_data, train_file, test_data, test_file):
    """
    Stores the training and test sets in the given files.
    :param train_data: The training set.
    :param train_file: The path to the training set file.
    :param test_data: The test set.
    :param test_file: The path to the test set file.
    :return:
    """
    train_data.to_csv(consts.DATA_PATH + train_file, index=None, header=None)
    test_data.to_csv(consts.DATA_PATH + test_file, index=None, header=None)


def show_histogram(training_set, test_set):
    """
    Shows a histogram of the data.
    :param training_set: The training set.
    :param test_set: The test set.
    """
    p1, p2 = 53, 32
    p1_train = training_set.loc[:, p1]
    p1_test = test_set.loc[:, p1]
    p2_train = training_set.loc[:, p2]
    p2_test = test_set.loc[:, p2]

    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].hist(p1_train, bins=20, color='blue', alpha=0.5, stacked=True, label='Train Feature')
    axs[0].hist(p1_test, bins=20, color='red', alpha=0.5, stacked=True, label='Test Feature')
    axs[0].legend()
    axs[0].set_title('Feature {}'.format(p1))
    axs[0].set_xlabel('Feature Values')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(p2_train, bins=20, color='blue', alpha=0.5, stacked=True, label='Train Feature')
    axs[1].hist(p2_test, bins=20, color='red', alpha=0.5, stacked=True, label='Test Feature')
    axs[1].legend()
    axs[1].set_title('Feature {}'.format(p2))
    axs[1].set_xlabel('Feature Values')
    axs[1].set_ylabel('Frequency')
    fig.show()


def train_with_lasso(dataset):
    """
    Trains the model.
    :param dataset: The dataset to train the model on.
    """
    y = np.array(dataset[81])
    dataset_copy = dataset.copy()
    dataset_copy.drop(81, axis=1, inplace=True)
    temp = np.array(dataset_copy)
    x = temp[:, 0:81]

    row, col = x.shape
    TR = round(row * 0.78)
    TT = row - TR

    x_train = x[0:TR, :]
    y_train = y[0:TR]

    classifier = Lasso(alpha=0.1)
    classifier.fit(x_train, y_train)

    x_test = x[TR:row, :]
    y_test = y[TR:row]

    y_pred = classifier.predict(x_test).round()

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    show_confusion_matrix(y_test, y_pred)
    display_confusion_matrix_statistics(cm, y_test, y_pred)


def show_confusion_matrix(y_true, y_pred):
    """
    Shows the confusion matrix.
    :param y_true: The actual values.
    :param y_pred: The predicted values.
    """
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classifier.classes_).plot()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()


def display_confusion_matrix_statistics(conf_matrix, y_true, y_pred):
    """
    Displays the statistics of the confusion matrix.
    :param conf_matrix: The confusion matrix.
    :param y_true: The actual values.
    :param y_pred: The predicted values.
    """

    num_classes = len(np.unique(y_true))



    if num_classes == 2:

        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TP = conf_matrix[1, 1]

        print("True Positive: {}".format(TP))
        print("True Negative: {}".format(TN))
        print("False Positive: {}".format(FP))
        print("False Negative: {}".format(FN))

        FPFN = FP + FN
        TPTN = TP + TN

        Accuracy = 1 / (1 + (FPFN / TPTN))
        print("Our_Accuracy_Score:", Accuracy)

        Precision = 1 / (1 + (FP / TP))
        print("Our_Precision_Score:", Precision)

        Sensitivity = 1 / (1 + (FN / TP))
        print("Our_Sensitivity_Score:", Sensitivity)

        Specificity = 1 / (1 + (FP / TN))
        print("Our_Specificity_Score:", Specificity)

        print("BuiltIn_Accuracy:", accuracy_score(y_true, y_pred))
        print("BuiltIn_Precision:", precision_score(y_true, y_pred))
        print("BuiltIn_Sensitivity (recall):", recall_score(y_true, y_pred))
    elif num_classes == 3:
        TP0 = conf_matrix[0, 0]
        TN0 = conf_matrix[1, 1] + conf_matrix[1, 2] + conf_matrix[2, 1] + conf_matrix[2, 2]
        FP0 = conf_matrix[0, 1] + conf_matrix[0, 2]
        FN0 = conf_matrix[1, 0] + conf_matrix[2, 0]

        print("Class 0 True Positive: {}".format(TP0))
        print("Class 0 True Negative: {}".format(TN0))
        print("Class 0 False Positive: {}".format(FP0))
        print("Class 0 False Negative: {}".format(FN0))

        FPFN = FP0 + FN0
        TPTN = TP0 + TN0

        Accuracy = 1 / (1 + (FPFN / TPTN))
        print("Class 0 Our_Accuracy_Score:", Accuracy)

        Precision = 1 / (1 + (FP0 / TP0))
        print("Class 0 Our_Precision_Score:", Precision)

        Sensitivity = 1 / (1 + (FN0 / TP0))
        print("Class 0 Our_Sensitivity_Score:", Sensitivity)

        Specificity = 1 / (1 + (FP0 / TN0))
        print("Class 0 Our_Specificity_Score:", Specificity)

        average_setting = 'micro'
        print("BuiltIn_Accuracy:", accuracy_score(y_true, y_pred))
        print("BuiltIn_Precision:", precision_score(y_true, y_pred, average=average_setting))
        print("BuiltIn_Sensitivity (recall):", recall_score(y_true, y_pred, average=average_setting))
    else:
        print("More than 3 classes or less than 2 classes not supported")


def train_with_random_forest(dataset):
    """
    Trains the Random Forest model.
    :param dataset: The dataset to train the model on.
    :return:
    """

    y = np.array(dataset[81])
    dataset_copy = dataset.copy()
    dataset_copy.drop(81, axis=1, inplace=True)
    temp = np.array(dataset_copy)
    x = temp[:, 0:81]

    row, col = x.shape
    TR = round(row * 0.78)
    TT = row - TR

    x_train = x[0:TR, :]
    y_train = y[0:TR]

    classifier = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
    model = classifier.fit(x_train, y_train)

    importance = model.feature_importances_
    indices = importance.argsort()[::-1]

    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    std = np.std([model.feature_importances_ for model in classifier.estimators_], axis=0)

    for f in range(dataset_copy.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))

    plt.bar(range(dataset_copy.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")

    plt.xticks(range(dataset_copy.shape[1]), indices + 1, rotation=90, fontsize=9)

    plt.show()

    oob_error = 1 - classifier.oob_score_

    x_test = x[TR:row, :]
    y_test = y[TR:row]

    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    display_confusion_matrix_statistics(cm, y_test, y_pred)
    show_confusion_matrix(y_test, y_pred)


def store_predicted_labels(dataset, y_pred, classifier):
    """
    Stores the predicted labels in the dataset.
    :param dataset: The dataset to store the predicted labels in.
    :param y_pred: The predicted labels.
    :param classifier: The classifier used to predict the labels.
    :return:
    """

    num_classes = len(np.unique(dataset[81]))

    dataset[82] = y_pred

    if classifier == Classifier.LASSO:
        filename = r"\01_lasso_predicted_labels.csv"
    elif classifier == Classifier.RANDOM_FOREST:
        if num_classes == 2:
            filename = r"\02_random_forest_predicted_labels.csv"
        elif num_classes == 3:
            filename = r"\012_random_forest_predicted_labels.csv"
    else:
        print("Classifier not supported")
        return

    file = consts.DATA_PATH + filename
    dataset.to_csv(file, index=False)


if __name__ == "__main__":
    main()
