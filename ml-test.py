import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import random
random.seed(0)

print('Choose a Dataset to work on')
print('''
\nEnter the number to choose the dataset
(1). Breast Cancer Dataset
(2). Digits Classification Dataset
(3). Flower Classification Dataset
''')
inp = int(input('>>> '))


if inp == 1:
    X,y = datasets.load_breast_cancer(return_X_y=True)
elif inp == 2:
    X,y = datasets.load_digits(return_X_y=True)
elif inp == 3:
    X,y = datasets.load_iris(return_X_y=True)
#print(X[0])


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,random_state=0,stratify=y,test_size = .3)

model_list = {
"knn": KNeighborsClassifier(),
"svm": SVC(),
"rfclf": RandomForestClassifier(),
"nn": MLPClassifier(max_iter = 5000),
}



for name,est in model_list.items():
    print(f'Using {name} model')
    est.fit(x_train,y_train)
    print('Score on Test set: ', est.score(x_test,y_test))
    #cv_mean_= sklearn.model_selection.cross_val_score(est,X,y,cv=7).mean()
    #print('Cross-validation mean score: ', cv_mean_)
    print()



print('Training Stopped 🛑') 