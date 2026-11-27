Naive Bayes Classifier

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.naive_bayes import GaussianNB

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('Naïve Bayes Classifier')

plt.tight_layout()

plt.show()

3 4 NAIVE BAYES CLASSIFIER 2. Simple Linear Regression

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)

X = np.linspace(1, 10, 100).reshape(-1, 1)

y = 10 + 8 * X[:, 0] + rng.normal(0, 4, len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Slope:', model.coef_[0], 'Intercept:', model.intercept_)

print('MSE:', mean_squared_error(y_test, y_pred))

print('R²:', r2_score(y_test, y_pred))

print('Predicted score for 6 study hours:', model.predict([[6]])[0])

plt.scatter(X_train[:, 0], y_train, label='Training data', alpha=0.5)

plt.scatter(X_test[:, 0], y_test, label='Test data')

plt.plot(X[:, 0], model.predict(X), color='red', label='Regression line')

plt.xlabel('Study hours')

plt.ylabel('Score')

plt.title('Simple Linear Regression')

plt.legend()

plt.show()

5 6 SIMPLE LINEAR REGRESSION 3. Multiple Linear Regression

import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(

 diabetes.data, diabetes.target, test_size=0.2, random_state=42

)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Intercept:', model.intercept_)

for feature, coefficient in zip(diabetes.feature_names, model.coef_):

 print(f'{feature}: {coefficient:.3f}')

print('MSE:', mean_squared_error(y_test, y_pred))

print('R²:', r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.7)

limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

plt.plot(limits, limits, 'r--', label='Perfect prediction')

plt.xlabel('Actual progression')

plt.ylabel('Predicted progression')

plt.title('Multiple Linear Regression')

plt.legend()

plt.show()

7 8 MULTIPLE LINEAR REGRESSION 4. Polynomial Regression

import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)

X = np.linspace(-5, 5, 120).reshape(-1, 1)

y = 2 + 3 * X[:, 0] + 1.5 * X[:, 0] ** 2 + rng.normal(0, 3, len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

random_state=42)

model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),

LinearRegression())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))

print('R²:', r2_score(y_test, y_pred))

plt.scatter(X_train[:, 0], y_train, alpha=0.5, label='Training data')

plt.scatter(X_test[:, 0], y_test, label='Test data')

plt.plot(X[:, 0], model.predict(X), 'r-', label='Degree 2 polynomial')

plt.xlabel('Input')

plt.ylabel('Target')

plt.title('Polynomial Regression')

plt.legend()

plt.show()

9 10 POLYNOMIAL REGRESSION 5. Logistic Regression

`import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,

random_state=42))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('Logistic Regression')

plt.tight_layout()

plt.show()

11 12 LOGISTIC REGRESSION 6. KNN

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

def KNN(new_point,data,K):

 distance = []

 #iterating over each row in data

 for row in data:

 # Calculating euclid distance of new_point from each row's feature

 dist = math.sqrt(((row[0]-new_point[0])**2)+((row[1]-

new_point[1])**2))

 distance.append([row[2],dist])

 # Sorting the distance in ascending order

 sorted_distance = sorted(distance, key=lambda row: row[1])

 

 #converting the sorted_distance nested list into df and only storing the 

elements upto K

 df = pd.DataFrame(sorted_distance[:K])

 voted = df[0].mode()

 print(voted)

 return df 

new_point = [5.3, 5.6]

KNN(new_point,data,12)

colors = {

 "Apple": "red",

 "Banana": "yellow",

 "Orange": "orange",

 "Pineapple": "green"

}

# Plot training data

for row in data:

 plt.scatter(row[0], row[1], color=colors[row[2]])

# Plot the new point

plt.scatter(

 new_point[0],

 new_point[1],

 color="black",

 marker="X", # Different marker

 s=200, # Bigger size

 edgecolors="white",

 linewidth=2,

 label="Test Point"

)

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

13 plt.legend()

plt.show()

Figure 1: KNN Plot

14 KNN 7. Decision Tree

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('Decision Tree Classification')

plt.tight_layout()

plt.show()

plt.figure(figsize=(14, 7))

plot_tree(model,

feature_names=iris.feature_names, class_names=list(iris.target_names),

 filled=True, rounded=True)

plt.title('Learned Decision Tree')

plt.tight_layout()

plt.show()

15 16 DECISION TREE 8. SVM Classification

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('SVM Classification')

plt.tight_layout()

plt.show()

17 18 SVM CLASSIFICATION 9. K Means Clustering

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('SVM Classification')

plt.tight_layout()

plt.show()

19 10. Artificial Neural Network

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,

ConfusionMatrixDisplay

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(

 iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target

)

model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8),

activation='relu', solver='lbfgs', max_iter=3000, random_state=42))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=iris.target_names,

zero_division=0))

print('Prediction for [5.1, 3.5, 1.4, 0.2]:',

 iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]])

ConfusionMatrixDisplay.from_predictions(y_test,

y_pred, display_labels=iris.target_names)

plt.title('Artificial Neural Network')

plt.tight_layout()

plt.show()

20 ARTIFICI
