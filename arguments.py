import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training ML model')
    parser.add_argument('--data', default=None, type=str)
    parser.add_argument('--methods', default = None, type=str, choices = ['Logistic_Regression','Linear_Regression','Naive_Bayes','SVM','K_Means','Random_Forest'])
    args = parser.parse_args()
    return args