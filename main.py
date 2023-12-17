from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np


raw_data = load_breast_cancer()
dane = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
X, y =load_breast_cancer(return_X_y=True, as_frame=True)
nazwy_kolumn = list(dane.columns)


#podział na zbiór testowy i uczący
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, shuffle=True)

#wykresy pudełkowe, żeby wiedzieć które dane przeskalować
import seaborn as sns
sns.boxplot(data = dane)#przyjmuje dataframe
sns.set(rc={'figure.figsize':(20,100)})#wielkosc (height,width)


#drzewo decyzyjne o wysokości 5 (nie może mieć przeskalowanych danych)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
model = DT(max_depth=5)#głębokość drzewa decyzyjnego
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)#można wypisać jako tablicę print(cm)
print("\nDrzewo decyzyjne(nieskalowane):")
print("Confusion matrix:",cm)
print("Czułość:",cm[0][0]/(cm[0][0]+cm[0][1]))
print("Precyzja:",cm[0][0]/(cm[0][0]+cm[1][0]))
print("Specyficzność:",cm[1][1]/(cm[1][1]+cm[1][0]))
print("Dokładność:",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))

from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
tree_vis = plot_tree(model,feature_names = nazwy_kolumn,
                     class_names=['N','Y'], fontsize = 10)

#skalowanie danych (standardowy skaler)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#wytrenowanie modelu do odrożniania nowotworów łagodnych od złośliwych
#klasyfikacja kNN
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix
model = kNN(n_neighbors=6)#ilosc branych pod uwagę sąsiadów "model = kNN(n_neighbors=4)"
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)#można wypisać jako tablicę print(cm)
print("\nKlasyfikacja kNN:")
print("Confusion matrix:",cm)
print("Czułość:",cm[0][0]/(cm[0][0]+cm[0][1]))
print("Precyzja:",cm[0][0]/(cm[0][0]+cm[1][0]))
print("Specyficzność:",cm[1][1]/(cm[1][1]+cm[1][0]))
print("Dokładność:",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))


#klasyfikacja SVC
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
model = SVM(kernel='poly', degree=8)#<-transformacja jądrowa "model = SVM()"
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)#można wypisać jako tablicę print(cm)
print("\nKlasyfikacja SVC:")
print("Confusion matrix:",cm)
print("Czułość:",cm[0][0]/(cm[0][0]+cm[0][1]))
print("Precyzja:",cm[0][0]/(cm[0][0]+cm[1][0]))
print("Specyficzność:",cm[1][1]/(cm[1][1]+cm[1][0]))
print("Dokładność:",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))




#wykonać SVM (jądro)

#%%
