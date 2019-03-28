import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score




url = "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data"  
names=['name','landmass','zone','area','population','language','religion','bars','strips','colors','red','green','blue','gold','white','black','orange','minhue','circles','cross','salt','quart','sun','cres','tri','icon','ani','text','top','bot']  
dataset = pd.read_csv(url, names=names)  
X = dataset.iloc[:, 1:29].values  
y = dataset.iloc[:, 29].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
lda = LDA(n_components=1)  
X_train = lda.fit_transform(X_train, y_train)  
X_test = lda.transform(X_test)  
classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  
cm = confusion_matrix(y_test, y_pred)  
print(cm)  
print('Accuracy' + str(accuracy_score(y_test, y_pred)))  


