from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image
import pandas as pd
import numpy as np

# 印出原始數據框的大小
df = pd.read_csv('/Users/anguschen/Desktop/Data/data/housing.csv')
print("原始大小:\n", df.shape, end="\n\n")


# 檢查數據框中是否存在缺失值（NAN值），並將包含缺失值的行印出
nan_rows = df[df.isnull().any(axis=1)]
print("有NAN值的row(前五行):\n", nan_rows, end="\n\n")


# 這部分程式碼將刪除包含缺失值的行，從而清理數據。然後它印出已刪除缺失值後的數據框的大小
df = df.dropna()
print("去除NAN值的大小:\n", df.shape, end="\n\n")


Y = df['ocean_proximity']
X = df.drop(['ocean_proximity'], axis=1)

# 數據劃分為訓練集(80%)和測試集(20%)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8, random_state=5)

# 創建一個決策樹
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)

# 使用訓練數據（X_train, Y_train）來訓練決策樹模型
tree.fit(X_train, Y_train)

excepted = Y_test

# 使用訓練模型（tree）對測試數據集（X_test）進行預測
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