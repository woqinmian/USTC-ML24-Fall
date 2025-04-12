from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np

iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df["target"] = list(iris.target)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]
# 划分数据

"""
在此填入你的代码
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sc = StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)


# 构建 mlp 模型
"""
在此填入你的代码
"""
mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=1000, random_state=42)
"""
在此填入你的代码
"""
mlp.fit(standard_train, Y_train)
# 得到预测结果
"""
在此填入你的代码
"""
result = mlp.predict(standard_test)


# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(result))
print("预测的准确率为：", mlp.score(standard_test, Y_test))
print("层数为：", mlp.n_layers_)
print("迭代次数为：", mlp.n_iter_)
print("损失为：", mlp.loss_)
print("激活函数为：", mlp.out_activation_)


# 代码的手动实现
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # sigmoid 计算方式
        """
        在此填入你的代码
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):  # sigmoid 导数计算方式
        """
        在此填入你的代码
        """
        return (1 - x) * x # 传入的x实际为sigmoid(x)

    def forward(self, X):
        """
        在此填入你的代码
        """
        self.input_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden # 隐藏层输入
        self.hidden_output = self.sigmoid(self.input_hidden) # 隐藏层输出

        self.input_output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output # 输出层输入
        self.output_output = self.sigmoid(self.input_output) # 输出层输出
        return self.output_output

    def backward(self, X, y, output, learning_rate):
        """
        在此填入你的代码
        """
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output) # 计算误差
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate # 修改参数

        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output) # 计算误差
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate # 修改参数

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (y - output) ** 2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))


# 将标签转换为独热编码
def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels


# 构建神经网络
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(Y_train))  # 根据训练集标签确定输出层大小
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 将标签转换为独热编码
Y_train_encoded = one_hot_encode(Y_train)

# 训练神经网络
print("training.......")
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.01)

# 预测测试集
predictions = nn.predict(standard_test)

# 计算准确率
accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)
