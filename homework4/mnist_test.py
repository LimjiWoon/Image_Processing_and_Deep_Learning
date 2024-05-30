import numpy as np
from nn.network import Network

from nn.flatten import Flatten

from nn.conv import Conv
from nn.fullyconnected import FullyConnected
from nn.activation import sigmoid, relu, mse, linear, cross_entropy
from nn.optimizer import AdamOptimizer, SGDOptimizer
import mnist_loader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split #4project

def accuracy(net, X, Y):
    a = (np.argmax(cross_entropy._softmax(net.forward(X)), axis=1) == np.argmax(Y, axis=1))
    return np.sum(a) / float(X.shape[0]) * 100.

def one_hot(x, size):
    a = np.zeros((x.shape[0], size))
    a[np.arange(x.shape[0]), x] = 1.
    return a


if __name__ == '__main__':

    ###########################################################################
    # TODO: 네트워크 초기화  (필요에 따라 내용을 수정후 레포트 작성)
    ###########################################################################

    # 심플 CNN 예제
    lr = 0.001
    layers = [
        Conv((5, 5, 1, 16), strides=1, activation=relu, optimizer=AdamOptimizer(),
             filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (28*28))),
        Conv((6, 6, 16, 32), strides=2, activation=relu, optimizer=AdamOptimizer(),
             filter_init=lambda shp:  np.random.normal(size=shp) * np.sqrt(1.0 / (16*24*24))),
        Conv((6, 6, 32, 64), strides=2, activation=relu, optimizer=AdamOptimizer(),
             filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (32*10*10))),
        Flatten((3, 3, 64)),
        FullyConnected((3*3*64, 256), activation=relu,
                       optimizer = AdamOptimizer(),
                       weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (3*3*64))),
        FullyConnected((256, 10), activation=linear,
                       optimizer = AdamOptimizer(),
                       weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (100.)))
    ]


    # 네트워크 객체 생성
    net = Network(layers, lr=lr, loss=cross_entropy)

    ###########################################################################
    # 데이터 가져오기
    ###########################################################################
    (train_data_X, train_data_Y), v, (tx, ty) = mnist_loader.load_data('./data/mnist.pkl.gz')
    train_data_Y = one_hot(train_data_Y, size=10)
    ty = one_hot(ty, size=10)
    train_data_X = np.reshape(train_data_X, [-1, 28, 28, 1])
    tx = np.reshape(tx, [-1, 28, 28, 1])

    print("data_X : " , train_data_X.shape)
    print("data_Y : " , train_data_Y.shape)

    # 데이터 분할하기
    train_data_X, val_data_X, train_data_Y, val_data_Y = train_test_split(
        train_data_X, train_data_Y, test_size=0.2, random_state=1234)

    print("train_data_X : " , train_data_X.shape)
    print("train_data_Y : " , train_data_Y.shape)
    print("val_data_X : " , val_data_X.shape)
    print("val_data_Y : " , val_data_Y.shape)


    ###########################################################################
    # TODO: 네트워크 학습  (필요에 따라 내용을 수정후 레포트 작성)
    ###########################################################################
    loss = []

    train_accuracy_history = []
    val_accuracy_history = []

    total_iter = 1000 # 학습의 반복 횟수
    batch_size = 50

    count = 0

    for iter in range(total_iter):
        shuffled_index = np.random.permutation(train_data_X.shape[0])

        batch_train_X = train_data_X[shuffled_index[:batch_size]]
        batch_train_Y = train_data_Y[shuffled_index[:batch_size]]
        net.train_step((batch_train_X, batch_train_Y))
        loss.append(np.sum(cross_entropy.compute(net.forward(batch_train_X), batch_train_Y)))

        if iter % 10 == 0:
            train_acc = accuracy(net, batch_train_X, batch_train_Y)
            val_acc = accuracy(net, val_data_X, val_data_Y)
            train_accuracy_history.append(train_acc)
            val_accuracy_history.append(val_acc)
            print(
                f'Iteration: {iter}, Loss: {loss[-1]:.6f}, Training Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}')
            if val_acc < max(val_accuracy_history):
                count += 1
                if count == 3:
                    total_iter = (iter+10)
                    break
            else:
                count = 0


    ###########################################################################
    # 마지막 결과 출력
    ###########################################################################
    print('#### 학습 종료 #####')
    print('Calculate accuracy over all test set (시간 소요)')
    test_acc = accuracy(net, tx, ty)
    print('Accuracy over all test set %.2f' % test_acc)
    print(total_iter)
    print(total_iter//10)
    plt.figure()
    plt.plot(range(total_iter//10), train_accuracy_history, label='Training Accuracy')
    plt.plot(range(total_iter//10), val_accuracy_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(range(total_iter-9), loss)
    plt.title('Test accuracy: %.2f' % test_acc)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['training loss'], loc='upper left')
    plt.show()

