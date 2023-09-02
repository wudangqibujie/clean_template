import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3}

# 训练模型
num_round = 20
bst = xgb.train(param, dtrain, num_round)

# 预测结果
preds = bst.predict(dtest)

# 计算准确率
accuracy = sum(1 for i in range(len(preds)) if preds[i] == y_test[i]) / float(len(preds))
print(f'Accuracy: {accuracy:.2f}')
