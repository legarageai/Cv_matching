from flask import Flask, render_template, request
#import fasttext
import PyPDF2
import pdfplumber
#import fastai
import nltk
import numpy as np
import pandas as pd
import os
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the 'punkt' resource
nltk.download('punkt')

# Charger le modèle FastText
#CV = './cc.fr.300.bin'
data_dir = './data/CV'
missing_skills_dir = './data/missing_skills'
#ft_model = fasttext.load_model(CV)

# Fonction pour calculer la similarité entre deux mots
def WordSimilarity(word1, word2):
    # embedding1 = ft_model.get_word_vector(word1)
    # embedding2 = ft_model.get_word_vector(word2)
    embedding1 = np.random.rand(300)  # Placeholder pour la similarité (à remplacer)
    embedding2 = np.random.rand(300)  # Placeholder pour la similarité (à remplacer)
    similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity_score

# Fonction pour calculer la similarité entre les compétences présentes et requises
def Similarity(present_skills, post_skills):
    n = 0
    for x in present_skills:
        List = []
        for y in post_skills:
            b = WordSimilarity(x, y)
            List.append(b)
        a = max(List)
        if a >= 0.5:
            n = n + 1
        elif 0.4 <= a < 0.5:
            n = n + 0.5
        else:
            n = n + 0
    similarity = n * 100 / len(post_skills)
    return similarity

# Charger les fichiers de compétences, domaines et diplômes
skills_def = pd.read_csv('./competences.csv', header=0)
skills_list = list(skills_def['etiquettes'])
domaines = pd.read_csv('./Domaines.csv', header=0)
diplomes = pd.read_csv('./Diplomes.csv', header=0)
diplomes_list = list(diplomes['Diplome'])
domaines_list = list(domaines['domaine'])

# Fonction pour extraire les compétences présentes dans un texte donné
def extract_present_skills(text):
    text_tokens = word_tokenize(text.lower())
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('french')]
    present_skills = []
    for skill in skills_list:
        if skill.lower() in tokens_without_sw:
            present_skills.append(skill)
    return list(set(present_skills))

# Fonction pour extraire les domaines présents dans un texte donné
def extract_present_domaines(text):
    present_domaines = []
    for domaine in domaines_list:
        if domaine.lower() in text.lower():
            present_domaines.append(domaine)
    return list(set(present_domaines))

# Fonction pour extraire les diplômes présents dans un texte donné
def extract_present_diplomes(text):
    present_diplomes = []
    for diplome in diplomes_list:
        if diplome.lower() in text.lower():
            present_diplomes.append(diplome)
    return list(set(present_diplomes))

app = Flask(__name__, template_folder='templates')

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Récupérez le fichier uploadé depuis la requête
    uploaded_file = request.files['file']
    # Vérifiez si un fichier a été téléchargé
    if uploaded_file is not None:
        # Vérifiez si le fichier PDF est valide
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        if num_pages > 0:
            # Extraction du texte de chaque page du fichier PDF
            script_req = []
            with pdfplumber.open(uploaded_file) as pdf:
                for i in range(num_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    script_req.append(text)

            # Prétraitement du texte
            nltk.download('stopwords')
            present_skills = extract_present_skills(' '.join(script_req))
            present_domaines = extract_present_domaines(' '.join(script_req))
            present_diplomes = extract_present_diplomes(' '.join(script_req))

            # Afficher les compétences, domaines et diplômes présents
            # print("Compétences présentes :", present_skills)
            # print("Domaines présents :", present_domaines)
            # print("Diplômes présents :", present_diplomes)
            

            # data_dir = './data/CV'
            # Vérifier si le dossier existe
            if os.path.exists(data_dir):
                # Liste des fichiers dans le répertoire
                files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

                # Dictionnaire pour stocker les scores
                scores_dict = {}

                # Parcourir les fichiers
                for file in files:
                    # Chemin d'accès complet du fichier
                    file_path = os.path.join(data_dir, file)

                    # Vérifier si le fichier est un PDF
                    if file.lower().endswith('.pdf'):
                        # Ouvrir le fichier PDF
                        with open(file_path, 'rb') as cv_file:
                            script = PyPDF2.PdfReader(cv_file)
                            num_pages = len(script.pages)
                            text_list = []

                            # Extraire le texte de chaque page du PDF
                            with pdfplumber.open(cv_file) as pdf:
                                for i in range(num_pages):
                                    page = pdf.pages[i]
                                    text = page.extract_text()
                                    text_list.append(text)

                            # Convertir la liste de textes en une seule chaîne de caractères
                            script = ' '.join(text_list)

                            # Appliquer le traitement NLP
                            #nltk.download('punkt')
                            #nltk.download('stopwords')

                            text_tokens = word_tokenize(script)

                            tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('french')]

                            script = ' '.join(tokens_without_sw)

                            # Extraire les compétences présentes dans le CV
                            cv_present_skills = extract_present_skills(script)

                            # Extraire les domaines présents dans le CV
                            cv_present_domaines = extract_present_domaines(script)

                            # Extraire les diplômes présents dans le CV
                            cv_present_diplomes = extract_present_diplomes(script)

                            # Calculer les similarités entre les compétences, domaines et diplômes
                            skills_similarity = Similarity(cv_present_skills, present_skills)
                            domaines_similarity = Similarity(cv_present_domaines, present_domaines)
                            diplomes_similarity = Similarity(cv_present_diplomes, present_diplomes)

                            # Calculer le score final
                            score_final = (domaines_similarity + diplomes_similarity + skills_similarity) / 3

                            # Stocker le score dans le dictionnaire
                            scores_dict[file] = score_final

                # Trier le dictionnaire des scores par ordre décroissant des valeurs
                sorted_scores = {k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)}

                # Créer une liste pour stocker les informations de chaque CV dans le classement
                cv_ranking = []

                # Parcourir les CV triés et récupérer les noms et scores
                for cv_name, score in sorted_scores.items():
                    cv_ranking.append((cv_name, score))
                    # Chemin d'accès complet du fichier de compétences manquantes
                    missing_skills_file = os.path.join(missing_skills_dir, cv_name + '.csv')

                    # Extraire les compétences manquantes dans le CV
                    missing_skills = list(set(present_skills) - set(cv_present_skills))

                    # Enregistrer les compétences manquantes dans un fichier CSV
                    with open(missing_skills_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Compétences'])
                        writer.writerows([[skill] for skill in missing_skills])

    # Si aucun fichier n'a été téléchargé ou s'il y a eu une erreur, renvoyez un message d'erreur
    return render_template('result.html', present_skills=present_skills, present_domaines=present_domaines, present_diplomes=present_diplomes, cv_ranking=cv_ranking)

@app.route('/cv_details', methods=['POST'])
def cv_details():
    cv_name = request.form['cv_name']
    action = request.form['action']

    cv_dir = data_dir
    cv_path = os.path.join(cv_dir, cv_name)

    if action == 'view':
    
            cv_content = ""

            if os.path.exists(cv_path):
                with open(cv_path, 'rb') as cv_file:
                    pdf_reader = PyPDF2.PdfReader(cv_file)
                    num_pages = len(pdf_reader.pages)
                    text_list = []

                    with pdfplumber.open(cv_file) as pdf:
                        for i in range(num_pages):
                            page = pdf.pages[i]
                            text = page.extract_text()
                            text_list.append(text)

                    cv_content = ' '.join(text_list)

            return render_template('cv_details.html', cv_content=cv_content)
        
    elif action == 'accept':
        status = 1
        # Chargement des données
        df_courses = pd.read_excel('final.xlsx')
        missing_skills_file = os.path.join(missing_skills_dir, cv_name + '.csv')
        df_missing_skills = pd.read_csv(missing_skills_file)

        # Création d'une matrice de similarité entre les compétences manquantes et les compétences des vidéos
        skills_corpus = df_courses['Compétences'].tolist()
        missing_skills_corpus = df_missing_skills['Compétences'].tolist()
        corpus = missing_skills_corpus + skills_corpus
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix[:len(missing_skills_corpus)], tfidf_matrix[len(skills_corpus):])
        # Fusionner les DataFrames df_missing_skills et df_courses en fonction de la colonne "Compétences"
        merged_df = pd.merge(df_missing_skills, df_courses, on='Compétences', how='left')

        # Recommandation des vidéos de formation et des certifications pour chaque compétence manquante
        recommended_videos = []
        recommended_certifications = []
        for i in range(len(merged_df)):
            missing_skill = merged_df.loc[i, 'Compétences']
            recommended_video = merged_df.loc[i, 'video']
            recommended_certification = merged_df.loc[i, 'certification']
            # Vérification pour éviter les valeurs "nan"
            if not pd.isnull(recommended_video):
                recommended_videos.append(recommended_video)
            if not pd.isnull(recommended_certification):
                recommended_certifications.append(recommended_certification)

        return render_template('cv_details.html', status=status, recommended_videos=recommended_videos, recommended_certifications=recommended_certifications)

    elif action == 'reject':
        status = 0
        # Chargement des données
        df_courses = pd.read_excel('final.xlsx')
        missing_skills_file = os.path.join(missing_skills_dir, cv_name + '.csv')
        df_missing_skills = pd.read_csv(missing_skills_file)

        # Création d'une matrice de similarité entre les compétences manquantes et les compétences des vidéos
        skills_corpus = df_courses['Compétences'].tolist()
        missing_skills_corpus = df_missing_skills['Compétences'].tolist()
        corpus = missing_skills_corpus + skills_corpus
        vectorizer = TfidfVectorizer() 
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix[:len(missing_skills_corpus)], tfidf_matrix[len(skills_corpus):])

        # Fusionner les DataFrames df_missing_skills et df_courses en fonction de la colonne "Compétences"
        merged_df = pd.merge(df_missing_skills, df_courses, on='Compétences', how='left')

        # Recommandation des vidéos de formation et des certifications pour chaque compétence manquante
        recommended_videos = []
        recommended_certifications = []
        for i in range(len(merged_df)):
            missing_skill = merged_df.loc[i, 'Compétences']
            recommended_video = merged_df.loc[i, 'video']
            recommended_certification = merged_df.loc[i, 'certification']
            # Vérification pour éviter les valeurs "nan"
            if not pd.isnull(recommended_video):
                recommended_videos.append(recommended_video)
            if not pd.isnull(recommended_certification):
                recommended_certifications.append(recommended_certification)
        return render_template('cv_details.html', status=status, recommended_videos=recommended_videos, recommended_certifications=recommended_certifications)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
