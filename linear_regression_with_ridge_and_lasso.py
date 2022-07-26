from sklearn.linear_model import LinearRegression , Lasso, Ridge,LogisticRegression
from sklearn import datasets,model_selection
import warnings,numpy as np
warnings.filterwarnings('ignore')
X,y = datasets.load_boston(return_X_y = True)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,random_state = 23,test_size = .3)

print('RIDGE')
ridge = Ridge(alpha=.01).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

print('LASSO')
lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# print('LOGISTIC')
# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))



