import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
df_courses = pd.read_excel('final.xlsx')
df_missing_skills = pd.read_csv('missing_skills.csv')

# Création d'une matrice de similarité entre les compétences manquantes et les compétences des vidéos de pip install --upgrade pip
skills_corpus = df_courses['Compétences'].tolist()  # Colonne contenant les compétences dans le fichier CSV des vidéos
missing_skills_corpus = df_missing_skills['Compétences'].tolist()
corpus = missing_skills_corpus + skills_corpus
tfidf_matrix = vectorizer.fit_transform(corpus)
similarity_matrix = cosine_similarity(tfidf_matrix[:len(missing_skills_corpus)], tfidf_matrix[len(skills_corpus):])

# Fusionner les DataFrames df_missing_skills et df_courses en fonction de la colonne "Compétences"
merged_df = pd.merge(df_missing_skills, df_courses, on='Compétences', how='left')

# Recommandation des vidéos de formation et des certifications pour chaque compétence manquante
for i in range(len(merged_df)):
    missing_skill = merged_df.loc[i, 'Compétences']
    recommended_video = merged_df.loc[i, 'video']
    recommended_certification = merged_df.loc[i, 'certification']
    print('\n Pour la compétence manquante "{}", voici la vidéo recommandée :\n{}'.format(missing_skill, recommended_video))
    print('\n Certification recommandée :\n {}'.format(recommended_certification))
