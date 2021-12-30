import numpy as np
from compute_utils import *
def Linear_Regression_GD(x,y,b,w,learning_rate=0.01,epochs=10000):
    '''
    설명 : gradient descent를 통해서 b,w를 epoch만큼 업데이트 시키고 return w,b 
    x : [bsz,representation]
    y : [bsz,]
    w : [representation]
    b : [1]
    '''
    bsz = y.shape[0]
    if w.ndim==1:
        w = np.reshape(w,(w.shape[0],1))
    if x.ndim==1:
        x = x.reshape(x.shape[0],1)
    if y.ndim==1:
        y = y.reshape(y.shape[0],1)
    for epoch in range(epochs):
        now_result = np.dot(x,w)+b    
        dw = 2*learning_rate*np.matmul((y-now_result).T,x)/bsz     #[1,representation]
        db = 2*learning_rate*(y-now_result).mean()
        w+=dw.T
        b+=db
        if epoch%1000==0:
            error = (y-now_result)**2/bsz
            print('err : ',np.mean(error))
    return w,b

def Linear_Regression(x,y,b,w,learning_rate=0.01,epochs=1000,batch_size=16):
    '''
    Linear Regression based on SGD

    input
        x       : [bsz,representation]
        y       : [bsz,1]
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''
    if w.ndim==1:
        w = np.reshape(w,(w.shape[0],1))
    if x.ndim==1:
        x = x.reshape(x.shape[0],1)
    if y.ndim==1:
        y = y.reshape(y.shape[0],1)
    for epoch in range(epochs):
        batch_mask = np.random.choice(x.shape[0],batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]

        now_result = np.dot(x_batch,w)+b 
        dw = 2*learning_rate*np.matmul((y_batch-now_result).T,x_batch)/batch_size     #[1,representation]
        db = 2*learning_rate*(y_batch-now_result).mean()
        w+=dw.T
        b+=db
        if epoch%1000==0:
            error = (y_batch-now_result)**2/batch_size
            print('err : ',np.mean(error))
    return w,b

def Logistic_Regression(x,y,number_of_class,learning_rate=0.0001,epochs=300000,batch_size = 32):
    '''
    Logistic Regression based on SGD

    input 
        x       : [data size,rep]
        y       : [data_size,1]   
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''
    w = np.random.uniform(-1,1,(x.shape[1],number_of_class))
    b = np.random.uniform(-1,1,(number_of_class))
    y = one_hot(y,number_of_class)
    for epoch in range(epochs):
        batch_mask = np.random.choice(x.shape[0],batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]
        z = sigmoid(x_batch.dot(w)+b)               # [bsz,1]
        dw = (y_batch-z).T.dot(x_batch)/batch_size  # [number of class,bsz]x[bsz,rep] = [number of class,rep]
        db = (y_batch-z).mean(axis=0)               # [1,number of class]
        w+=learning_rate*dw.T
        b+=learning_rate*db
    return w,b

def SVM(x,y,number_of_class,C=30,learning_rate=0.001,epochs=10000):
    '''
    이진 linear SVM만을 가정
    input
        x       : [data size,rep]
        y       : [data size,1]
    parameters
        w       : [rep,number of class]
        b       : [1, number of class]
    output
        w       : [rep,number of class]
        b       : [1, number of class]
    '''
    w = np.ones((x.shape[1]))
    b = np.ones((1))
    
    for i in range(epochs):
        distance = y*(np.dot(x,w)+b)-1
        distance[np.where(distance>0)]=0                     # margin위에 있지 않은 data의 distance=0 처리-> margin위에 있는 a=1이라고 처리한 효과와 동일
        L = 1/2*np.dot(w,w)-C*np.sum(distance)
        dw = np.zeros(len(w))
        db = np.zeros(len(b))
        for idx, d in enumerate(distance):
            if d==0:
                dL_w=w
                dL_b=0
            else:
                dL_w = w - C*y[idx]*x[idx]
                dL_b = - C*y[idx]
            dw+=dL_w
            db+=dL_b
        w = w-learning_rate*dw/x.shape[0]
        b = b-learning_rate*db/x.shape[0]
    return w,b

def Naive_Bayes(x,y):
    '''
    input
        x : [data size, rep]
        y : [data size, 1]
    output
        p_rep_status    : y의 상태에 따라 rep=1일 확률
        p_pos           : y=1인 데이터 발생 확률
    '''
    # pre-processing
    number_of_class = len(collections.Counter(y))

    # discretize
    x_threshold = x.mean(axis=0)       
    for idx,rep in enumerate(x):
        x[idx] = rep>x_threshold
    x = x.astype(np.int64)

    # build classifier
    number_of_positives = sum(y)
    number_of_negatives = len(y)-number_of_positives
    p_pos = number_of_positives/y.shape[0]
    p_rep_status = np.zeros((number_of_class,x.shape[1]))
    
    for data,status in zip(x,y):
        for idx,rep in enumerate(data):
            if status==1 and rep==1:
                p_rep_status[status][idx]+=1
            elif status==0 and rep==1:
                p_rep_status[status][idx]+=1
    p_rep_status+=1
    p_rep_status[1]/=(number_of_positives+number_of_class)
    p_rep_status[0]/=(number_of_negatives+number_of_class)
    return x_threshold,p_rep_status,p_pos

def K_means():
    pass

def Random_Forest():
    pass

def EM_Algorithm():
    pass