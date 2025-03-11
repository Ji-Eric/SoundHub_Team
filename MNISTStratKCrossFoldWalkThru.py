#imports
#not good at this whole bagel truck yet
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#First fetch from openml, x shape (7000,784), y(70000) and pixle values are in x, digit labels 0-9 in y
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.to_numpy() # shape:(70000, 784)
y = y.to_numpy().astype(int)  #string labels like '5' to ints
print("Full dataset shape:", X.shape, y.shape)
#Start k-fold#######
#doing 3 splits, split size is ~46666 training, ~23334 testing
#essentially, stratified part of this, is that if entire data set has say ~10% images labeled at 8, in each of our folds, we'll 
#get around ~10% of data trained on images labeled as 8
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
accuracy_scores = []
fold_number = 1
#Cross validation
for train_idx, test_idx in skf.split(X, y):
    print(f"\n~~~ Fold {fold_number} ~~~~~~~~~~")
    fold_number += 1
    #extracting train/test data for curr fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Traiing size: {X_train.shape[0]}, Testsize: {X_test.shape[0]}")
    #train classifier reduce n_estimators if too slow
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  #brrr
    )
    print("loading, wait")
    clf.fit(X_train, y_train)
    #get accuracy scores###
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    print(f"Accuracy on for {fold_number} number fold: {acc:.4f}")
#Get average performances
print("\nCross-validation complete.")
print("Fold accuracies:", accuracy_scores)
print(f"Mean accuracy across folds: {np.mean(accuracy_scores):.4f}")

