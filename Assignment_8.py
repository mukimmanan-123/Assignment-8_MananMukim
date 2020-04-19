import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from seaborn import heatmap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Reading the data-set
data_set = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(data_set.head())

# Check For Null Values
print(data_set.isnull().sum())

# For Duplicate Columns if present
print(len(data_set))
print(len(data_set.drop_duplicates()))

# Checking For Unique Values
print(data_set["BusinessTravel"].unique())
print(data_set["EmployeeCount"].unique())
print(data_set["StandardHours"].unique())
print(data_set["Over18"].unique())

# Removing Columns
columns_to_remove = ["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"]
data_set.drop(columns=columns_to_remove, inplace=True)

"""Data-Set Visualization"""

# Variations Of Data In respective columns
data_set.hist()
fig = plt.gcf()
fig.set_size_inches(20, 20)

# Correlation
cor_mat = data_set.corr()
mask = np.array(cor_mat)
# Taking the lower triangle
mask[np.tril_indices_from(mask)] = False
heatmap(data_set.corr(), annot=True, mask=mask, cbar=True)
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.show()

# KDE
sns.kdeplot(data_set['Age'], color='purple')
plt.show()

sns.kdeplot(data_set['TotalWorkingYears'], color='black')
plt.show()

sns.kdeplot(data_set['YearsAtCompany'], color='purple')
plt.show()

sns.kdeplot(data_set['PercentSalaryHike'], color='purple')
plt.show()

# Variations
plt.hist(data_set['Attrition'], bins=3)
plt.show()

plt.hist(data_set['BusinessTravel'], bins=5)
plt.show()

plt.hist(data_set['OverTime'], bins=3)
plt.show()

plt.hist(data_set['Department'], bins=5)
plt.show()

plt.hist(data_set['EducationField'], bins=11)
fig = plt.gcf()
fig.set_size_inches(10, 5)

plt.hist(data_set['Gender'], bins=3)
plt.show()

plt.hist(data_set['JobRole'], bins=17)
fig = plt.gcf()
fig.set_size_inches(20, 10)

# Encoding Columns
data_set.loc[:, "Attrition"] = LabelEncoder().fit_transform(data_set.loc[:, "Attrition"])
data_set.loc[:, "BusinessTravel"] = LabelEncoder().fit_transform(data_set.loc[:, "BusinessTravel"])
data_set.loc[:, "Department"] = LabelEncoder().fit_transform(data_set.loc[:, "Department"])
data_set.loc[:, "EducationField"] = LabelEncoder().fit_transform(data_set.loc[:, "EducationField"])
data_set.loc[:, "Gender"] = LabelEncoder().fit_transform(data_set.loc[:, "Gender"])
data_set.loc[:, "JobRole"] = LabelEncoder().fit_transform(data_set.loc[:, "JobRole"])
data_set.loc[:, "MaritalStatus"] = LabelEncoder().fit_transform(data_set.loc[:, "MaritalStatus"])
data_set.loc[:, "OverTime"] = LabelEncoder().fit_transform(data_set.loc[:, "OverTime"])

# Data-Set Columns
print(data_set.columns)

# Data-Set Extract
X = data_set.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]].values
y = data_set.iloc[:, 1].values

# Scaling The Values
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
X_scaled = np.append(arr=np.ones((len(X_scaled), 1)).astype(float), values=X_scaled, axis=1)

# Splitting Data-Set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=0.4)

# OverSampling
Smote = SMOTE(random_state=0)
X_train_Over, y_train_Over = Smote.fit_resample(X_train, y_train)

# UnderSampling
NearMiss = NearMiss()
X_train_Under, y_train_Under = NearMiss.fit_sample(X_train, y_train)

# Scoring
scoring = make_scorer(balanced_accuracy_score)

# Logistic Regression
def Logistic_Grid():
    parameter = [{'penalty': ["l2", "none"]}]
    return parameter


def Decision_Grid():
    parameter = [{'criterion': ["gini", "entropy"]}]
    return parameter


def Random_Grid():
    parameter = [{'criterion': ["gini", "entropy"],
                  'n_estimators': [100, 200, 300, 400, 500]}]
    return parameter


def K_NN_Grid():
    parameter = [{'n_neighbors': [3, 5, 7]}]
    return parameter


def SVM_Grid():
    parameter = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                 {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
                  'gamma': [0.1, 0.001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    return parameter


def Scores_And_GridSearch(string, value):
    global parameters
    if value == 0:
        x, y_ = X_train, y_train
    elif value == 1:
        x, y_ = X_train_Over, y_train_Over
    else:
        x, y_ = X_train_Under, y_train_Under

    if string == "Logistic":
        parameters = Logistic_Grid()
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring=scoring,
                                   cv=10,
                                   n_jobs=-1)

        grid_search = grid_search.fit(x, y_)
        best_parameters = grid_search.best_params_
        print(best_parameters)
    elif string == "Decision":
        parameters = Decision_Grid()
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring=scoring,
                                   cv=10,
                                   n_jobs=-1)

        grid_search = grid_search.fit(x, y_)
        best_parameters = grid_search.best_params_
        print(best_parameters)
    elif string == "Random":
        parameters = Random_Grid()
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring=scoring,
                                   cv=10,
                                   n_jobs=-1)

        grid_search = grid_search.fit(x, y_)
        best_parameters = grid_search.best_params_
        print(best_parameters)
    elif string == "K":
        parameters = K_NN_Grid()
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring=scoring,
                                   cv=10,
                                   n_jobs=-1)

        grid_search = grid_search.fit(x, y_)
        best_parameters = grid_search.best_params_
        print(best_parameters)
    elif string == "SVM":
        parameters = SVM_Grid()
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring=scoring,
                                   cv=10,
                                   n_jobs=-1)

        grid_search = grid_search.fit(x, y_)
        best_parameters = grid_search.best_params_
        print(best_parameters)


def score_calculator():
    print("Accuracy :", balanced_accuracy_score(y_test, predictions))
    print("Confusion metric :", confusion_matrix(y_test, predictions))
    print("f1_Score :", f1_score(y_test, predictions))
    print("Precision :", precision_score(y_test, predictions))
    print("Recall :", recall_score(y_test, predictions))


# Logistic Classifier (Original Sample)
print("Logistic Classifier (Original Sample)")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Logistic", 0)

# Logistic Classifier (Original Sample) With Parameter Tuned
print("Logistic Classifier (Original Sample) With Tuned Parameter")
classifier = LogisticRegression(penalty='none')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# Logistic Classifier (Over Sampling)
print("=" * 40)
print("Logistic Classifier (Over Sampling)")
classifier = LogisticRegression()
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Logistic", 1)

# Logistic Classifier (Over Sampling) With Parameter Tuned
print("=" * 40)
print("Logistic Classifier (Over Sampling) With Parameter Tuned")
classifier = LogisticRegression(penalty='l2')
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# Logistic Classifier (Under Sampling)
print("=" * 40)
print(" Logistic Classifier (Under Sampling)")
classifier = LogisticRegression()
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Logistic", 2)

# Logistic Classifier (Under Sampling) With Parameter Tuned
print("=" * 40)
print(" Logistic Classifier (Under Sampling) With Parameter Tuned")
classifier = LogisticRegression(penalty='l2')
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()

# Naive Bayes Classifier (Original Sample)
print("=" * 40)
print("Naive Bayes Classifier (Original Sample)")
classifier = GaussianNB()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# Naive Bayes Classifier (Over Sampling)
print("=" * 40)
print("Naive Bayes Classifier (Over Sampling)")
classifier = GaussianNB()
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# Naive Bayes Classifier (Under Sampling)
print("=" * 40)
print("Naive Bayes Classifier (Under Sampling)")
classifier = GaussianNB()
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()

# Decision Tree Classifier (Original Sample)
print("=" * 40)
print("Decision Tree Classifier (Original Sample)")
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Decision", 0)

# Decision Tree Classifier (Original Sample) with parameter Tuned
print("=" * 40)
print("Decision Tree Classifier (Original Sample) With Parameter Tuned")
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# Decision Tree Classifier (Over Sampling)
print("=" * 40)
print("Decision Tree Classifier (Over Sampling)")
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Decision", 1)

# Decision Tree Classifier (Over Sampling) With Parameter Tuned
print("=" * 40)
print("Decision Tree Classifier (Over Sampling), With Parameter Tuned")
classifier = DecisionTreeClassifier(random_state=0, criterion='gini')
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# Decision Tree Classifier (Under Sampling)
print("=" * 40)
print("Decision Tree Classifier (Under Sampling)")
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Decision", 2)

# Decision Tree Classifier (Under Sampling) With Parameter Tuned
print("=" * 40)
print("Decision Tree Classifier (Under Sampling) With Parameter Tuned")
classifier = DecisionTreeClassifier(random_state=0, criterion='entropy')
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()

# Random Forest Classifier (Original Sample)
print("=" * 40)
print("Random Forest Classifier (Original Sample)")
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Random", 0)

# Random Forest Classifier (Original Sample) With Parameter Tuned
print("=" * 40)
print("Random Forest Classifier (Original Sample) With Parameter Tuned")
classifier = RandomForestClassifier(criterion='gini', n_estimators=300)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# Random Forest Classifier (Over Sampling)
print("=" * 40)
print("Random Forest Classifier (Over Sampling)")
classifier = RandomForestClassifier()
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Random", 1)

# Random Forest Classifier (Over Sampling) With Parameter Tuned
print("=" * 40)
print("Random Forest Classifier (Over Sampling) With Parameter Tuned")
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100)
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# Random Forest Classifier (Under Sampling)
print("=" * 40)
print("Random Forest Classifier (Under Sampling)")
classifier = RandomForestClassifier()
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("Random", 2)

# Random Forest Classifier (Under Sampling) With Parameter Tuned
print("=" * 40)
print("Random Forest Classifier (Under Sampling) With Parameter Tuned")
classifier = RandomForestClassifier(criterion='entropy', n_estimators=200)
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()

# K-NN Classifier (Original Data)
print("=" * 40)
print("K-NN Classifier (Original Data)")
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("K", 0)

# K-NN Classifier (Original Data) With Parameter Tuned
print("=" * 40)
print("K-NN Classifier (Original Data) With Parameter Tuned")
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# K-NN Classifier (Over Sampling)
print("=" * 40)
print("K-NN Classifier (Over Sampling)")
classifier = KNeighborsClassifier()
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("K", 1)

# K-NN Classifier (Over Sampling) With Parameter Tuned
print("=" * 40)
print("K-NN Classifier (Over Sampling) with Parameter Tuned")
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# K-NN Classifier (Under Sampling)
print("=" * 40)
print("K-NN Classifier (Under Sampling)")
classifier = KNeighborsClassifier()
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()

# Support Vector Classifier (Original Sample)
print("=" * 40)
print("Support Vector Classifier (Original Sample)")
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("SVM", 0)

# Support Vector Classifier (Original Sample) With Parameter Tuned
print("=" * 40)
print("Support Vector Classifier (Original Sample) With Parameter Tuned")
classifier = SVC(kernel='rbf', C=1000, gamma=0.01)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score_calculator()

# Support Vector Classifier (Over Sampling)
print("=" * 40)
print("Support Vector Classifier (Over Sampling)")
classifier = SVC(kernel='rbf')
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("SVM", 1)

# Support Vector Classifier (Over Sampling) With Parameter Tuned
print("=" * 40)
print("Support Vector Classifier (Over Sampling) With Parameter Tuned")
classifier = SVC(kernel='rbf', C=10, gamma=0.2)
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()

# Support Vector Classifier (Under Sampling)
print("=" * 40)
print("Support Vector Classifier (Under Sampling)")
classifier = SVC(kernel='rbf')
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("SVM", 2)

# Support Vector Classifier (Under Sampling) With Parameter Tuned
print("=" * 40)
print("Support Vector Classifier (Under Sampling) With Parameter Tuned")
classifier = SVC(kernel='rbf', C=1, gamma=0.1)
classifier.fit(X_train_Under, y_train_Under)
predictions = classifier.predict(X_test)
score_calculator()
Scores_And_GridSearch("SVM", 2)

"""After Performing parameter tuning, over-sampling and under-sampling we came to a conclusing for choosing
a model with good recall_score and a good balanced_accuracy score.
We are choosing balanced accuracy score because it is a measure of recall of positive class + recall of negative class 
and it outperforms f1_score when positives >> negatives
We according to the results got, the best model is Logistic Classifier with oversampling
Cause we are getting a good balanced accuracy around 76%
And a recall about 76.5%"""

print("=" * 40)
print("Logistic Classifier")
classifier = LogisticRegression(penalty='l2')
classifier.fit(X_train_Over, y_train_Over)
predictions = classifier.predict(X_test)
score_calculator()
