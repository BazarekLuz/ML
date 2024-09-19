import numpy as np
from matplotlib import pyplot as plt, patches
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.base import clone
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# # download the MNIST image
mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target


# let's see the number
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')


# some_digit = X[0]
# plot_digit(some_digit)
# plt.show()
#
# # this data set has already divided data into train and test sets
# # this is an example of binary classification - only two classes: 5s and non-5s
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# y_train_5 = (y_train == '5')
# y_test_5 = (y_test == '5')
#
# sgd_classifier = SGDClassifier(random_state=42)
# sgd_classifier.fit(X_train, y_train_5)
#
# # check accuracy with cross validation
# print(cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring="accuracy"))
#
# # dummy classifier, check non-5
# dummy_classifier = DummyClassifier()
# dummy_classifier.fit(X_train, y_train_5)
# print(any(dummy_classifier.predict(X_train)))  # it returns false, so it does not find any 5s
#
# # check accuracy for finding non-5s
# print(cross_val_score(dummy_classifier, X_train, y_train_5, cv=3, scoring="accuracy"))
#
# # testing accuracy is not the best technique to measure performance of classifiers
# # confusion matrix is the one to go
#
# # custom cross validation
# skfolds = StratifiedKFold(n_splits=3)
#
# # split data to train and test sets and implement cross validation
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_classifier)
#
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))
#
# y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
# # confusion matrix
# cm = confusion_matrix(y_train_5, y_train_pred)
# print(cm)
#
# # pretend to have perfect predictions, to see perfect confusion matrix
# y_train_perfect_predictions = y_train_5
# print(confusion_matrix(y_train_5, y_train_perfect_predictions))
#
# # some of the measures to the confusion matrix - precision, recall
# # precision = True Positive / (True Positive + False Positive)
# # recall = True Positive / (True Positive + False Negative)
# print(precision_score(y_train_5, y_train_pred))
# print(recall_score(y_train_5, y_train_pred))
#
# # f1 score is another indicator to measure quality of classificator it contains both of precision and recall
# print(f1_score(y_train_5, y_train_pred))
#
# # get result of decision function for samples
# y_scores = sgd_classifier.decision_function([some_digit])
# print(y_scores)
#
# threshold = 3000
# y_some_digit_pred = (y_scores > threshold)
# print(y_some_digit_pred)
#
# # define threshold for predictions
# y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method="decision_function")
#
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#
# plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
# plt.plot(thresholds, recalls[:-1], 'g-', label="Recall", linewidth=2)
# plt.vlines(threshold, 0, 1.0, 'k', 'dotted', label='Threshold')
#
# # extra code – this section just beautifies and saves Figure 3–5
# idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
# plt.plot(thresholds[idx], precisions[idx], "bo")
# plt.plot(thresholds[idx], recalls[idx], "go")
# plt.axis([-50000, 50000, 0, 1])
# plt.grid()
# plt.xlabel("Threshold")
# plt.legend(loc="center right")
#
# plt.show()
#
# # another figure for precision/recall ratio
# plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
#
# plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
#
# # extra code – just beautifies and saves Figure 3–6
# plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
# plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
# plt.plot([recalls[idx]], [precisions[idx]], "ko",
#          label="Point at threshold 3,000")
# plt.gca().add_patch(patches.FancyArrowPatch(
#     (0.79, 0.60), (0.61, 0.78),
#     connectionstyle="arc3,rad=.2",
#     arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
#     color="#444444"))
# plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.axis([0, 1, 0, 1])
# plt.grid()
# plt.legend(loc="lower left")
#
# plt.show()
#
# # find first argument to have >= 0.90 precision
# idx_for_90_precision = (precisions >= 0.90).argmax()
# threshold_for_90_precision = thresholds[idx_for_90_precision]
# print(threshold_for_90_precision)
#
# y_train_pred_90 = (y_scores > threshold_for_90_precision)
# print(precision_score(y_train_5, y_train_pred_90))
# # 0.9
# recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
# print(recall_at_90_precision)
# # 0.48
# # precision is usually not useful with classification, so this is a bad result
#
# # ROC
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#
# idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
# tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
#
# plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
# plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
# plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
# plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
#
# # extra code – just beautifies and saves Figure 3–7
# plt.gca().add_patch(patches.FancyArrowPatch(
#     (0.20, 0.89), (0.07, 0.70),
#     connectionstyle="arc3,rad=.4",
#     arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
#     color="#444444"))
# plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
# plt.xlabel('False Positive Rate (Fall-Out)')
# plt.ylabel('True Positive Rate (Recall)')
# plt.grid()
# plt.axis([0, 1, 0, 1])
# plt.legend(loc="lower right", fontsize=13)
#
# plt.show()
#
# # AUC - area under curve - score the ROC
# print(roc_auc_score(y_train_5, y_scores))
#
# # use different classifier - Random Forest Classifier - to compare results
# forest_classifier = RandomForestClassifier(random_state=42)
#
# y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3, method='predict_proba')
# print(y_probas_forest[:2])
#
# y_scores_forest = y_probas_forest[:, 1]
# precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
#
# plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
#
# plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
#          label="Random Forest")
# plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
#
# # extra code – just beautifies and saves Figure 3–8
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.axis([0, 1, 0, 1])
# plt.grid()
# plt.legend(loc="lower left")
#
# plt.show()
#
# # random forest has much better results than previous classifier
# y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
# print(f1_score(y_train_5, y_train_pred_forest))
# print(roc_auc_score(y_train_5, y_train_pred_forest))
# print(precision_score(y_train_5, y_train_pred_forest))
# print(recall_score(y_train_5, y_train_pred_forest))
#
# # ----------------------------------------------------------------------------------------------------------------------
# # non-binary classification
#
# svm_classifier = SVC(random_state=42)
# svm_classifier.fit(X_train[:2000], y_train[:2000])
# print(svm_classifier.predict([some_digit]))
#
# some_digit_scores = svm_classifier.decision_function([some_digit])
# print(some_digit_scores.round(2))
#
# # scale train set
# sgd_classifier = SGDClassifier(random_state=42)
# sgd_classifier.fit(X_train, y_train)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
# print(cross_val_score(sgd_classifier, X_train_scaled, y_train, cv=3, scoring='accuracy'))
#
# y_train_pred = cross_val_predict(sgd_classifier, X_train_scaled, y_train, cv=3)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
# plt.show()
#
# # multilabel classification
# y_train_large = (y_train >= '7')
# y_train_odd = (y_train.astype('int8') % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]
#
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_multilabel)
#
# print(knn.predict([some_digit]))
# y_train_knn_pred = cross_val_predict(knn, X_train, y_multilabel, cv=3)
# print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))
#
# chain_classifier = ClassifierChain(SVC(), cv=3, random_state=42)
# chain_classifier.fit(X_train[:2000], y_multilabel[:2000])
#
# print(chain_classifier.predict([some_digit]))

# multi output classification
np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = y_test

knn = KNeighborsClassifier()
knn.fit(X_train_mod, y_train_mod)
clean_digit = knn.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
