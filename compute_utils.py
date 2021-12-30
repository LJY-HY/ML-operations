import numpy as np
import collections
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    '''
    input  : [bsz,c]
    output : [bsz,c]
    '''
    max_a = a.max()
    a-=max_a
    exp_a = np.exp(a)
    if a.ndim==1:
        # input차원이 1차원인 경우
        sum_a = np.sum(exp_a)
        return exp_a/sum_a
    else:
        # input차원이 1차원 이상인 경우
        sum_a = np.sum(exp_a,axis=1).reshape(np.sum(exp_a,axis=1).shape[0],1)
        return exp_a/sum_a

def one_hot(y, k):
    '''
    input 
        y : [bsz,1]
        k : [1,]
    
    output
        y_oh : [bsz,k]
    '''
    labels = collections.Counter(y)
    if min(labels.keys())==0:
        # 주어진 y가 0부터 시작하는 경우
        y_oh = np.eye(k)[y]
    else:
        # 주어진 y가 1부터 시작하는 경우
        y_oh = np.eye(k)[y-1]

    return y_oh


def cross_entropy_loss(y,t):
    '''
    input
        y : [bsz, label]           label들은 one-hot 형태인지 아닌지는 모름
        t : [bsz, # of class]
    '''
    number_of_class = t.shape[-1]
    encoding_matrix = np.eye(number_of_class,dtype=int)
    mu = 10**(-6)
    logits = softmax(t)
    if y.ndim==1:                                       # bsz가 1인경우
        if y.shape[0]==1:                               # one-hot이 아닌 경우
            y = encoding_matrix[y]                      # [number of class]
        return -np.sum(y*np.log(logits+mu))
    else:
        bsz = t.shape[0]
        if y.shape[1]==1:                               # one-hot이 아닌 경우
            y = encoding_matrix[y]                      # [bsz, number of class]
        return -np.sum(y*np.log(logits+mu))/bsz

def divide_dataset(x,y):
    train_mask = np.random.choice(x.shape[0],int(x.shape[0]*0.8),replace=False)
    test_mask = np.delete(np.arange(len(x)),train_mask)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]
    return x_train,y_train,x_test,y_test


