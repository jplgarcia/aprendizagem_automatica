import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import explained_variance_score, matthews_corrcoef, max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr


# Load the dataset
df = pd.read_csv("HA1-DatasetScaled.tsv", sep="\t")

print("::::::::::::::::::::::::::::")
print(":::::::: REGRESSION ::::::::")
print("::::::::::::::::::::::::::::")

X_supercoductivity_df = df.drop("critical_temp", axis=1)
yc_supercoductivity_df = df["critical_temp"]
X_supercoductivity = X_supercoductivity_df.to_numpy()
yc_supercoductivity = yc_supercoductivity_df.to_numpy()

# Separating an Independent Validation Set (IVS).
X_TRAIN, X_IVS, y_TRAIN, y_IVS = train_test_split(X_supercoductivity, yc_supercoductivity, test_size=0.25, random_state=1337)

max_depth_values = [3, 5, 10, 30]
# In each max depth
for max_depth in max_depth_values:
    # Decision Tree Regression with K-fold Cross-Validation
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=23)
    kf_reg.get_n_splits(X_TRAIN, y_TRAIN)
    TRUTH_nfold=None
    PREDS_nfold=None
    for train_index, test_index in kf_reg.split(X_TRAIN):
        X_train, X_test = X_TRAIN[train_index], X_TRAIN[test_index]
        y_train, y_test = y_TRAIN[train_index], y_TRAIN[test_index]
        
        mdl = DecisionTreeRegressor(max_depth=max_depth)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        if TRUTH_nfold is None:
            PREDS_nfold=preds
            TRUTH_nfold=y_test
        else:
            PREDS_nfold=np.hstack((PREDS_nfold, preds))
            TRUTH_nfold=np.hstack((TRUTH_nfold, y_test))

    print("\n:::::::::::::: MAX DEPTH " + str(max_depth) + " :::::::::::::::")
    print("The RVE is: ", explained_variance_score(y_test, preds))
    print("The rmse is: ", mean_squared_error(y_test, preds, squared=False))
    corr, pval=pearsonr(y_test, preds)
    print("The Correlation Score is is: %6.4f (p-value=%e)"%(corr,pval))
    print("The Maximum Error is is: ", max_error(y_test, preds))
    print("The Mean Absolute Error is: ", mean_absolute_error(y_test, preds))

print("\n\n\n")

print("::::::::::::::::::::::::::::")
print(":::::::: CLASSIFIER ::::::::")
print("::::::::::::::::::::::::::::")

df["binary_target"] = df["critical_temp"] >= 80.0
X_supercoductivity_df = df.drop("critical_temp", axis=1)
X_supercoductivity_df = X_supercoductivity_df.drop("binary_target", axis=1)
yc_supercoductivity_df = df["binary_target"]

X_supercoductivity = X_supercoductivity_df.to_numpy()
yc_supercoductivity = yc_supercoductivity_df.to_numpy()

# Separating an Independent Validation Set (IVS).
X_TRAIN, X_IVS, y_TRAIN, y_IVS = train_test_split(X_supercoductivity, yc_supercoductivity, test_size=0.25, random_state=1337)

max_depth_values = [3, 5, 10, 30]
## In each max depth
for max_depth in max_depth_values:
    # Decision Tree Classification with K-fold Cross-Validation
    kf_class = KFold(n_splits=5, shuffle=True, random_state=23)
    kf_class.get_n_splits(X_TRAIN, y_TRAIN)
    TRUTH_nfold=None
    PREDS_nfold=None
    for train_index, test_index in kf_class.split(X_TRAIN):
        X_train, X_test = X_TRAIN[train_index], X_TRAIN[test_index]
        y_train, y_test = y_TRAIN[train_index], y_TRAIN[test_index]
        
        mdl = DecisionTreeClassifier(max_depth=max_depth)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        if TRUTH_nfold is None:
            PREDS_nfold=preds
            TRUTH_nfold=y_test
        else:
            PREDS_nfold=np.hstack((PREDS_nfold, preds))
            TRUTH_nfold=np.hstack((TRUTH_nfold, y_test))

    print("\n:::::::::::::: MAX DEPTH " + str(max_depth) + " :::::::::::::::")
    print("The Precision is: %7.4f" % precision_score(TRUTH_nfold, PREDS_nfold))
    print("The Recall is: %7.4f" % recall_score(TRUTH_nfold, PREDS_nfold))
    print("The F1 score is: %7.4f" % f1_score(TRUTH_nfold, PREDS_nfold))
    print("The Matthews correlation coefficient is: %7.4f" % matthews_corrcoef(TRUTH_nfold, PREDS_nfold))

