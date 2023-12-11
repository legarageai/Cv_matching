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
from sklearn.metrics.pairwise import cosine_similarity
import spacy # Librairie for NLP word entity
from spacy.lang.fr.stop.word import STOP_WORDS

nlp = spacy.load('fr_core_news_sm')  # Load the French language model

# Charger le modèle FastText
#CV = './cc.fr.300.bin'
data_dir = './data/CV'
missing_skills_dir = './data/missing_skills'
#ft_model = fasttext.load_model(CV)

# Fonction pour calculer la similarité entre deux mots avec Jaccard(Code update)
def WordSimilarity(word1, word2):
    # embedding1 = ft_model.get_word_vector(word1)
    # embedding2 = ft_model.get_word_vector(word2)
    embedding1 = np.random.rand(300)  # Placeholder pour la similarité (à remplacer)
    embedding2 = np.random.rand(300)  # Placeholder pour la similarité (à remplacer)
    
    # Cosine Similarity 
    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Jaccard similarity
    set1 = set(word1)
    set2 = set(word2)
    
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set1.union(set2)))
    
    jaccard_similarity = intersection_size / union_size if union_size !=0 else 0
    
    # Combine cosine and Jaccard similarity 
    combined_similarity = 0.7 * cosine_similarity + 0.3 * jaccard_similarity
    
    return combined_similarity

# Fonction pour calculer la similarité entre les compétences présentes et requises(Code update)
def calculate_similarity(present_skills, post_skills):
    
    total_similarity = 0
    
    for present_skill in present_skills:
        similarity_score = []
        
        for post_skill in post_skills:
            similarity = WordSimilarity(present_skill, post_skill)
            similarity_score.append(similarity)
            
        max_similiraty = max(similarity_score)
        
        if max_similiraty >= 0.5:
            similarity_score += 1
        elif 0.4 <= max_similiraty < 0.5:
            total_similarity += 0.5
            
        if len(post_skills) >= 0:
            similarity_percentage = (total_similarity * 100) / len(post_skills)
        else:
            similarity_percentage = 0
            
    return similarity_percentage

# Charger les fichiers de compétences, domaines et diplômes
skills_def = pd.read_csv('./competences.csv', header=0)
skills_list = list(skills_def['etiquettes'])
domaines = pd.read_csv('./Domaines.csv', header=0)
diplomes = pd.read_csv('./Diplomes.csv', header=0)
diplomes_list = list(diplomes['Diplome'])
domaines_list = list(domaines['domaine'])

# Fonction pour extraire les compétences présentes dans un texte donné(Code Update)
def extract_present_skills(text):
     # Process the text with spaCy NLP
    doc = nlp(text.lower())
    
    # Extract tokens without stop words
    tokens_without_sw = [token.text for token in doc if token.text not in STOP_WORDS]
    
    #text_tokens = word_tokenize(text.lower())
    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('french')]
    
    
    present_skills = []
    for ent in doc.ents:
        # Check if the entity label is relevant to skills (you can customize this condition)
        if ent.label_ == 'SKILL' and ent.text.lower() in tokens_without_sw:
            present_skills.append(ent.text)

    return list(set(present_skills))

    #for skill in skills_list:
        #if skill.lower() in tokens_without_sw:
            #present_skills.append(skill)
    #return list(set(present_skills))

# Fonction pour extraire les domaines présents dans un texte donné(Code Update)
def extract_present_domaines(text):
    # Process the text with spaCy NLP
    doc = nlp(text)
    
    present_domaines = set()
    #for domaine in domaines_list:
     #   if domaine.lower() in text.lower():
      #      present_domaines.append(domaine)
      
    for ent in doc.ents:
        if ent.text.lower() in domaines_list:
            present_domaines.add(ent.text.lower())
            
    return list(present_domaines)

# Fonction pour extraire les diplômes présents dans un texte donné
def extract_present_diplomes(text):
    # Process the text with spaCy NLP
    doc = nlp(text)
    
    present_diplomes = set()
    
    for ent in doc.ents:
        if ent.text.lower() in diplomes_list:
            present_diplomes.add(ent.text.lower())
            
    #for diplome in diplomes_list:
     #   if diplome.lower() in text.lower():
      #      present_diplomes.append(diplome)
    return list(present_diplomes)

# Download the 'punkt' resource
nltk.download('punkt')

app = Flask(__name__, template_folder='templates')

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')


# This route host the function upload for PDF file(Code update)
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Récupérez le fichier uploadé depuis la requête
        uploaded_file = request.files['file']
    
        # Vérifiez si un fichier a été téléchargé
        if uploaded_file is not None and uploaded_file.filename != '':        
            # Vérifiez si le fichier a une extension PDF
            if uploaded_file.filename.lower().endswith('.pdf'):
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
                                    skills_similarity = calculate_similarity(cv_present_skills, present_skills)
                                    domaines_similarity = calculate_similarity(cv_present_domaines, present_domaines)
                                    diplomes_similarity = calculate_similarity(cv_present_diplomes, present_diplomes)

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
                else:
                    return render_template('error.html', error_message="Le fichier PDF est vide.")
            else:
                return render_template('error.html', error_message="Le fichier n'est pas au format PDF.")
        else:
            return render_template('error.html', error_message="Aucun fichier n'a été téléchargé.")
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    
# This route host the function which can show the CV details(Code reviewS)
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

        # df_missing_skills = pd.read_csv(missing_skills_file)
        df_missing_skills = pd.read_csv(missing_skills_file, encoding='iso-8859-1')


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

        # df_missing_skills = pd.read_csv(missing_skills_file)
        df_missing_skills = pd.read_csv(missing_skills_file, encoding='iso-8859-1')


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
