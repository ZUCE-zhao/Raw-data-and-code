from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# 构建KNN模型流水线，包含标准化处理和网格搜索
pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())


# 读取数据并跳过第一行表头
path = r"E:\codes\line_force\data"
filename = 'data2.csv'
with open(path + '\\' + filename) as temp_f:
    # get No of columns in each line
    col_count = [len(l.split(",")) for l in temp_f.readlines()]
column_names = [i for i in range(max(col_count))]
data = pd.read_csv(path + '\\' + filename, skiprows=1, skip_blank_lines=True, header=None, names=column_names)
print(column_names)

print("病态步态数据规模：", data.shape)
X = data.iloc[:, :6]
# y = data.iloc[:, 7].apply(lambda x: 1 if x > 183 else (2 if x < 177 else 0))  # 三分类任务
# y = data.iloc[:, 7].apply(lambda x: 1 if x > 185 else 0) #81%
y = data.iloc[:, 7].apply(lambda x: 1 if x > 183 else 0)

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 使用GridSearchCV进行参数搜索
param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最佳参数和训练
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 预测
y_pred = best_model.predict(X_test)

# 评估模型性能
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
