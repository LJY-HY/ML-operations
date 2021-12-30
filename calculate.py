from compute_utils import *
from arguments import get_arguments
from methods import *
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer
import argparse
import numpy
import collections

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()

    if args.methods=='Linear_Regression':
        # y = 2x+5
        x = np.array([2,6,9])
        y = np.array([9,17,23])
        w = np.random.uniform(-1,1,1)
        b = np.random.uniform(-1,1,1)
        w,b = globals()[args.methods](x,y,w,b)
        print('w:',w)
        print('b:',b)
    elif args.methods=='Logistic_Regression':
        x,y = load_iris(return_X_y=True)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)

        w,b = globals()[args.methods](x_train,y_train,number_of_class)

        print('accuracy : {:.4f}%'.format(100*sum(y_test==np.argmax(x_test.dot(w)+b,axis=1))/y_test.shape[0]))
    elif args.methods=='SVM':
        x,y = make_blobs(n_samples=150,centers=2,random_state=20)
        y = np.sign(y-0.5).astype(np.int64)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)

        w,b = globals()[args.methods](x_train,y_train,number_of_class)

        print('accuracy : {:.4f}%'.format(100*sum(y_test == np.sign(x_test.dot(w)+b).astype(np.int64))/x_test.shape[0]))
    elif args.methods=='Naive_Bayes':
        data = load_breast_cancer()
        x = data['data']
        y = data['target']
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        
        x_threshold,p_rep_status,p_pos = globals()[args.methods](x_train,y_train)

        # prediction
        prediction = np.zeros((x_test.shape[0]))
        for idx, rep in enumerate(x_test):
            x_test[idx] = rep>x_threshold
        x_test = x_test.astype(np.int64)

        for idx,data in enumerate(x_test):
            prod_p_pos=1
            prod_p_neg=1
            # calculate p(y=1|x)
            for j in range(x_test.shape[1]):
                prod_p_pos*=(p_rep_status[1][j]*data[j]+(1-p_rep_status[1][j])*(1-data[j]))
                prod_p_neg*=(p_rep_status[0][j]*data[j]+(1-p_rep_status[0][j])*(1-data[j]))
            prediction[idx] = prod_p_pos*p_pos / (prod_p_pos*p_pos+prod_p_neg*(1-p_pos))
        print('accuracy : {:.4f}%'.format(100*sum(y_test == (prediction>0.5).astype(np.int64))/x_test.shape[0]))
    elif args.methods =='K_means':
        pass
if __name__=='__main__':
    main()

