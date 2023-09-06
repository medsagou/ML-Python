import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# Charger les données
data = pd.read_excel('./data2.xlsx')
data
# Effectuer One-Hot Encoding
data.columns = data.columns.str.strip() # pour supprimer s'il y a des cases vides

data = pd.get_dummies(data, columns=['zone'])
data

# Diviser les données en ensembles d'entraînement et de test
X = data.drop(['Nomre de départ'], axis=1)
y_regression = data['Nomre de départ']
y_classification = data.drop(['Nomre de départ'], axis=1)

X_train, X_test, y_regression_train, y_regression_test, y_classification_train, y_classification_test = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42
)

# Entraîner le modèle de régression pour prédire le nombre de départs
regression_model = LinearRegression()
regression_model.fit(X_train, y_regression_train)

# Prévoir les départs pour l'année 2022
y_regression_pred = regression_model.predict(X_test)

# Calculer le R² et le RMSE pour la régression
r2 = r2_score(y_regression_test, y_regression_pred)
rmse = mean_squared_error(y_regression_test, y_regression_pred, squared=False)

print(f"R² (Régression) : {r2}")
print(f"RMSE (Régression) : {rmse}")

# Entraîner le modèle de classification pour prédire le type des zones
classification_model = RandomForestClassifier()
classification_model.fit(X_train, y_classification_train)

# Prévoir le type des zones
y_classification_pred = classification_model.predict(X_test)

# Calculer l'exactitude pour la classification
accuracy = accuracy_score(y_classification_test, y_classification_pred)

print(f"Exactitude (Classification) : {accuracy}")
prediction_zone = classification_model.predict(X_test)
prediction_zone

# Exemple de prévision pour l'année 2022 et le type des zones
zone_A = 0
zone_B = 1
zone_C = 0
zone_D = 1
nouveaux_departs = regression_model.predict([[zone_A, zone_B, zone_C, zone_D]])
nouveau_type_zone = classification_model.predict([[zone_A, zone_B, zone_C,zone_D]])

print(f"Prévision Départs 2022 : {nouveaux_departs}")
print(f"Prévision Type de Fonctionnement : {nouveau_type_zone}")
#--------------------------------