import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
df_courses = pd.read_excel('final.xlsx')
df_missing_skills = pd.read_csv('missing_skills.csv')

# Création d'une matrice de similarité entre les compétences manquantes et les compétences des vidéos
skills_corpus = df_courses['Compétences'].tolist()
missing_skills_corpus = df_missing_skills['Compétences'].tolist()
corpus = missing_skills_corpus + skills_corpus
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
similarity_matrix = cosine_similarity(tfidf_matrix[:len(missing_skills_corpus)], tfidf_matrix[len(skills_corpus):])

# Fusionner les DataFrames df_missing_skills et df_courses en fonction de la colonne "Compétences"
merged_df = pd.merge(df_missing_skills, df_courses, on='Compétences', how='left')

# Affichage des recommandations avec Streamlit
for i in range(len(merged_df)):
    missing_skill = merged_df.loc[i, 'Compétences']
    recommended_video = merged_df.loc[i, 'video']
    recommended_certification = merged_df.loc[i, 'certification']
    
    st.write(f"Pour la compétence manquante '{missing_skill}', voici la vidéo recommandée :")
    st.write(recommended_video)
    st.write("Certification recommandée :")
    st.write(recommended_certification)
    st.write("---")
