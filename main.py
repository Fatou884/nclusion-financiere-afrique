import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import joblib

#importation du fichier
data = pd.read_csv('C:/Users/ashin/PycharmProjects/Financial_inclusion.csv')

print(data.info())

# Afficher des statistiques descriptives
print(data.describe())

# Gérer les valeurs manquantes
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())


# Remplacer les valeurs manquantes dans les colonnes catégorielles par le mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

#suppresion doublons
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)

#gérer les valeurs aberants
# Boxplot pour la taille des ménages
sns.boxplot(data['household_size'])
plt.show()

# Boxplot pour l'âge des répondants
sns.boxplot(data['age_of_respondent'])
plt.show()

# Suppression des valeurs aberrantes (si nécessaire)
Q1 = data['household_size'].quantile(0.25)
Q3 = data['household_size'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['household_size'] < (Q1 - 1.5 * IQR)) | (data['household_size'] > (Q3 + 1.5 * IQR)))]

# Encodage des variables catégorielles
data_encoded = pd.get_dummies(data)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Séparer les caractéristiques et la cible
X = data_encoded.drop('target', axis=1)
y = data_encoded['target']

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import streamlit as st

# Charger le modèle
import joblib
model = joblib.load('model.pkl')

# Créer l'application Streamlit
st.title("Prédiction d'utilisation de compte bancaire")

# Ajouter des champs de saisie
age = st.number_input("Âge", min_value=18, max_value=100)
income = st.number_input("Revenu")
# Ajouter d'autres champs selon les fonctionnalités de votre modèle

# Bouton de validation
if st.button('Faire une prédiction'):
    features = [[age, income]]  # Remplir avec les autres fonctionnalités
    prediction = model.predict(features)
    st.write("Probabilité d'utilisation d'un compte bancaire : ", prediction)

