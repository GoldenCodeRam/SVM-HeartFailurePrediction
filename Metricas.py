import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv ('Data.csv', sep = ',')
X = data.drop('HeartDisease', axis=1) # caracteristicas
Y = data['HeartDisease'] # clase 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=508) # 20% para test
clf = SVC() # instancia el algoritmo
clf.fit(X_train, y_train) #se entrena

y_pred = clf.predict(X_test)

## MATRIZ DE CONFUSION

matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)


## METRICAS

VP=matriz[0][0]
VN=matriz[1][1]
FP=matriz[1][0]
FN=matriz[0][1]
#Exactitud
accuracy=((VP+VN)/((VP+VN)+(FN+FP)))
#Sensibilidad
sensibility=(VP/(VP+FN))
#Precision
precision=(VP/(VP+FP))
#Puntuación f1
f1=2*((precision*sensibility)/(precision+sensibility))

print('Accuracy: ', accuracy*100)
print('Sensibility: ', sensibility*100)
print('Precision: ', precision*100)
print('F1 Score: ', f1)


## PREDICCIONES

"""

49,1,2,160,180,0,0,156,0,1.0,1,1
49,0,3,140,234,0,0,140,1,1.0,1,1
42,1,2,115,211,0,1,137,0,0.0,2,0
54,1,1,120,273,0,0,150,0,1.5,1,0
54,1,2,130,294,0,1,100,1,0.0,1,1
35,0,1,150,264,0,0,168,0,0.0,2,0
"""



falsos = [[[42,1,2,115,211,0,1,137,0,0.0,2]], [[54,1,1,120,273,0,0,150,0,1.5,1]],[[35,0,1,150,264,0,0,168,0,0.0,2]]]
positivos = [[[49,1,2,160,180,0,0,156,0,1.0,1]], [[49,0,3,140,234,0,0,140,1,1.0,1]], [[54,1,2,130,294,0,1,100,1,0.0,1]]]

for f in falsos:
    print(f, 'Prediccion: ' + str(clf.predict(f)), 'Pertenece a 0')
for v in positivos:
    print(f, 'Prediccion: ' + str(clf.predict(v)), 'Pertenece a 1')

