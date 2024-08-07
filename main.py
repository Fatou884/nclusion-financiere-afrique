import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib

# Importation du fichier
data = pd.read_csv('Financial_inclusion.csv')


# Gérer les valeurs manquantes
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Remplacer les valeurs manquantes dans les colonnes catégorielles par le mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Suppression des doublons
data.drop_duplicates(inplace=True)


# Suppression des valeurs aberrantes (si nécessaire)
Q1 = data['household_size'].quantile(0.25)
Q3 = data['household_size'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['household_size'] < (Q1 - 1.5 * IQR)) | (data['household_size'] > (Q3 + 1.5 * IQR)))]

# Séparer les caractéristiques et la cible avant l'encodage
X = data.drop('bank_account', axis=1)
y = data['bank_account']

# Encodage des variables catégorielles
X_encoded = pd.get_dummies(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Entraînement du classifieur
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = clf.predict(X_test)


# Sauvegarder le modèle
joblib.dump(clf, 'model.pkl')

# Charger le modèle
model = joblib.load('model.pkl')

# Créer l'application Streamlit
st.title("Prédiction d'utilisation de compte bancaire")

# Ajouter des champs de saisie
country = st.text_input("Pays")
year = st.number_input("Année", min_value=1900, max_value=2100)
uniqueid = st.text_input("ID unique")
bank_account = st.selectbox("Compte bancaire", ["Oui", "Non"])
location_type = st.selectbox("Type de lieu", ["Rural", "Urbain"])
cellphone_access = st.selectbox("Accès au téléphone portable", ["Oui", "Non"])
household_size = st.number_input("Taille du ménage", min_value=1)
age_of_respondent = st.number_input("Âge du répondant", min_value=18)
gender_of_respondent = st.selectbox("Genre du répondant", ["Homme", "Femme"])
relationship_with_head = st.selectbox("Relation avec le chef de ménage", ["Conjoint", "Enfant", "Autre"])
marital_status = st.selectbox("État civil", ["Célibataire", "Marié", "Divorcé", "Veuf"])
education_level = st.selectbox("Niveau d'éducation", ["Aucun", "École primaire", "École secondaire", "Université"])
job_type = st.selectbox("Type d'emploi", ["Agriculteur", "Employé", "Ouvrier", "Autre"])

# Bouton de validation
if st.button('Faire une prédiction'):
    features = pd.DataFrame([[country, year, uniqueid, bank_account, location_type, cellphone_access,
                              household_size, age_of_respondent, gender_of_respondent,
                              relationship_with_head, marital_status, education_level, job_type]],
                            columns=['country', 'year', 'uniqueid', 'bank_account', 'location_type',
                                     'cellphone_access', 'household_size', 'age_of_respondent',
                                     'gender_of_respondent', 'relationship_with_head', 'marital_status',
                                     'education_level', 'job_type'])
    features_encoded = pd.get_dummies(features)

    # Align with model features
    features_encoded = features_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    prediction = model.predict(features_encoded)
    st.write("Probabilité d'utilisation d'un compte bancaire : ", prediction[0])



