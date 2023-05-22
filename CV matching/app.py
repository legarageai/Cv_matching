import streamlit as st
import PyPDF2
import pdfplumber
import fasttext
import nltk
import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfFileReader

# Charger le modèle FastText
CV = './cc.fr.300.bin'
ft_model = fasttext.load_model(CV)

# Fonction pour calculer la similarité entre deux mots
def WordSimilarity(word1, word2): 
    embedding1 = ft_model.get_word_vector(word1)
    embedding2 = ft_model.get_word_vector(word2)
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

# Afficher la zone de téléchargement de fichier
uploaded_file = st.file_uploader("Ajoutez une fiche descriptive de poste au format PDF", type="pdf")

# Vérifier si un fichier a été téléchargé
if uploaded_file is not None:
    # Lecture du fichier PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)

    st.write(f"Nombre de pages : {num_pages}")

    # Extraction du texte de chaque page du fichier PDF
    script_req = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i in range(num_pages):
            page = pdf.pages[i]
            text = page.extract_text()
            script_req.append(text)

    # Prétraitement du texte
    #nltk.download('stopwords')
    present_skills = extract_present_skills(' '.join(script_req))
    present_domaines = extract_present_domaines(' '.join(script_req))
    present_diplomes = extract_present_diplomes(' '.join(script_req))

    # Afficher les compétences, domaines et diplômes présents
    st.write("Compétences présentes :", present_skills)
    st.write("Domaines présents :", present_domaines)
    st.write("Diplômes présents :", present_diplomes)

    data_dir = './data/CV'
    # Renseigner le dossier contenant les CV
    #data_dir = st.text_input("Entrez le chemin d'accès du dossier contenant les CV")


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

                    # Extraire les compétences de la fiche de poste qui ne sont pas dans le CV
                    missing_skills = []
                    for skill in post_skills:
                        if skill not in present_skills:
                            missing_skills.append(skill)
                    #missing_skills = list(set(missing_skills))
                    print(missing_skills)

                    # Générer le chemin d'accès pour le fichier CSV des compétences manquantes
                    csv_filename = os.path.splitext(file)[0] + ".csv"
                    file_path = os.path.join('skills_manquant', csv_filename)

                    with open(file_path, 'w', newline='') as files:
                        writer = csv.writer(files)
                        writer.writerow(['Compétences manquantes'])
                        for skill in missing_skills:
                            writer.writerow([skill])

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

        # Afficher le classement
        st.write("Classement des CV :")
        for index, (file, score) in enumerate(sorted_scores.items(), 1):
            st.write(f"{index}. {file} | Score : {score}")
