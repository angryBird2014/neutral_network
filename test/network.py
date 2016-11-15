import numpy as np
import random
import pickle
import gzip
import matplotlib.pyplot as pl

def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    train_data,validation_data,test_data = pickle.load(f,encoding="latin1")
    f.close()
    return (train_data,validation_data,test_data)


def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    train_input = [np.reshape(x,(784,1))for x in tr_d[0]]
    train_result = [vectorized_result(y) for y in tr_d[1]]
    train_data = zip(train_input,train_result)
    validate_input = [np.reshape(x,[784,1]) for x in va_d[0]]
    validate_data = zip(validate_input,va_d[1])
    test_input = [np.reshape(x,[784,1]) for x in te_d[0]]
    test_data = zip(test_input,te_d[1])
    return (train_data,validate_data,test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmod_prime(z):
    return sigmod(z)*(1-sigmod(z))

def drawLoss(data):
    pl.plot(data[0],data[1])
    pl.show()

class network(object):

    def __init__(self,size):
        self.size = size
        self.layer_number = len(size)
        self.bias = [np.random.randn(y,1)
                     for y in size[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(size[:-1],size[1:])]
    def test(self):
        print(self.bias)
        print("--------**********------")
        print(self.weights)


    def SGD(self,training_data,epochs,mini_batch_size,eta):
        '''
        :param training_data:训练数据
        :param epochs:迭代次数
        :param mini_batch_size:小批量大小
        :param eta:学习速率
        :return:null
        '''
        n = len(training_data)
        loss_data=[]

        for j in range(epochs):
            loss_ = 0
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batchs:
                loss_ += self.update_mini_batch(mini_batch,eta)
            loss_data.append(loss_/mini_batch_size)

        drawLoss(zip(loss_data,range[epochs]))




    def update_mini_batch(self,mini_batch,eta):
        '''
        在小批度样本上运用梯度下降
        :param mini_batch:随机梯度下降集大小
        :param eta:学习速率
        :return:
        '''
        loss=0
        nable_b = [np.zeros(b.shape) for b in self.bias]            #临时变量
        nable_w = [np.zeros(w.shape) for w in self.weights]         #临时变量
        for x,y in mini_batch:
            loss_sum = 0
            delta_nable_w,delta_nbale_b ,activations= self.backprop(x,y)
            loss_sum += np.linalg.norm(y,activations[-1],2)         #2范数
            nable_b = [nb+dnb for nb,dnb in zip(nable_b,delta_nbale_b)]
            nable_w = [nw+dnw for nw,dnw in zip(delta_nable_w,nable_w)]
        loss = loss_sum/len(mini_batch)
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nable_w)]
        self.bias = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.bias,nable_b)]
        return loss

    def backprop(self,x,y):
        nable_b = [np.zeros(b.shape) for b in self.bias]  # 临时变量
        nable_w = [np.zeros(w.shape) for w in self.weights]  # 临时变量
        #feedforward
        activation = x
        activations = [x]
        zs =[]
        for b,w in zip(self.bias,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmod(z)
            activations.append(activation)
        #backpass
        delta = self.cost_detrivation(activations[-1],y) * sigmod_prime(zs[-1])
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.layer_number):
            z = zs[-l]
            delta = (self.weights[-l+1].transpose())*sigmod_prime(z)
            nable_b[-l] = delta
            nable_w[-l] = np.dot(activations[-l-1],delta)
        return (nable_b,nable_w,activations)

    def cost_detrivation(self,out_activiton,y):
        return (out_activiton-y)

if __name__ == '__main__':
    size=[2,3,4,5]
    neutral_nwtwork = network(size)
    #neutral_nwtwork.test()
    train,validate,test= load_data_wrapper()
    print(train)