from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image
import pandas as pd
import numpy as np

# Print the size of the original dataframe
df = pd.read_csv('/Users/anguschen/Desktop/Data/data/housing.csv')
print("原始大小:\n", df.shape, end="\n\n")


# Check for missing values (NAN values) in the dataframe and print rows with missing values
nan_rows = df[df.isnull().any(axis=1)]
print("有NAN值的row(前五行):\n", nan_rows, end="\n\n")


# This part of the code will remove rows with missing values, cleaning the data. Then it prints the size of the dataframe after removing missing values
df = df.dropna()
print("去除NAN值的大小:\n", df.shape, end="\n\n")


Y = df['ocean_proximity']
X = df.drop(['ocean_proximity'], axis=1)

# Split the data into training set (80%) and test set (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8, random_state=5)

# Create a decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)

# Train the decision tree model using the training data (X_train, Y_train)
tree.fit(X_train, Y_train)

excepted = Y_test

# Use the trained model (tree) to predict the test dataset (X_test)
predicted = tree.predict(X_test)

print("混淆矩陣:\n", metrics.confusion_matrix(excepted, predicted), end="\n\n")
print("分類報告:\n", metrics.classification_report(excepted, predicted), end="\n\n")


dot_data = StringIO()
export_graphviz(tree, out_file = dot_data,
                feature_names = X.columns,
                filled = True, rounded = True,
                special_characters = True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#print("graph.write")
#Image(graph.create_png())

graph.write_png("decision_tree.png")

print("graph write complete")
