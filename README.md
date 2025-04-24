# SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset.Convert it into a pandas DataFrame and separate features X and labels y.
2. Divide the data into training and test sets using train_test_split.
3. Initialize the SGDClassifier with a maximum number of iterations.Fit the model on the training data.
4. Predict labels for the test set.Calculate accuracy score and display the confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Prajin S
RegisterNumber:  212223230151
*/
```
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
df.head()
df.tail()
X=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

sgd_clf.fit(x_train,y_train)

y_pred=sgd_clf.predict(x_test)

y_pred
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy : {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(cm)
```

## Output:

![image](https://github.com/user-attachments/assets/88636a6a-9af3-438c-89c7-765a95d5e393)

![image](https://github.com/user-attachments/assets/47a0306f-6bd0-4f3e-a51f-55cdeaa2ca01)

![image](https://github.com/user-attachments/assets/23479814-5f5c-4999-891f-3d683ba51384)

![image](https://github.com/user-attachments/assets/ebbcc238-d1f4-420b-a862-4e4f05b60181)

![image](https://github.com/user-attachments/assets/a401ff51-8b10-4ca1-845f-eba8ca7d0433)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
