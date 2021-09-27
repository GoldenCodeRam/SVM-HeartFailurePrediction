import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# LECTURA DE ARCHIVO CSV (DATA SET)
def getSex(x):
    if x == 'M': return 0
    elif x == 'F': return 1

def getChestPainType(x):
    if x == 'TA': return 0
    elif x == 'ATA': return 1
    elif x == 'NAP': return 2
    elif x == 'ASY': return 3

def getRestingECG(x):
    if x == 'Normal': return 0
    elif x == 'ST': return 1
    elif x == 'LVH': return 2

def getExerciseAngina(x):
    if x == 'N': return 0
    elif x == 'Y': return 1

def getST_Slope(x):
    if x == 'Down': return 0
    elif x == 'Flat': return 1
    elif x == 'Up': return 2


data = pd.read_csv ('heart.csv', sep = ',')

data["Sex"] = data["Sex"].apply(getSex)
data["ChestPainType"] = data["ChestPainType"].apply(getChestPainType)
data["RestingECG"] = data["RestingECG"].apply(getRestingECG)
data["ExerciseAngina"] = data["ExerciseAngina"].apply(getExerciseAngina)
data["ST_Slope"] = data["ST_Slope"].apply(getST_Slope) 

print(data.head())


#____________IMPLEMENTACIÓN DEL ALGORITMO___________________________________

X = data.drop('HeartDisease', axis=1) # caracteristicas
Y = data['HeartDisease'] # clase 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20) # 20% para test
clf = SVC() # instancia el algoritmo
clf.fit(X_train, y_train) #se entrena
score = clf.score(X_test, y_test)  # se prueba
print(score)    