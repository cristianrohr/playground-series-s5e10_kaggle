from sklearn.metrics import root_mean_squared_error

def evaluate_model(yreal: list, ypred: list):
    return root_mean_squared_error(yreal, ypred)