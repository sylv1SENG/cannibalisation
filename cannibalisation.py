import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from serpapi import GoogleSearch

# Fonction pour récupérer le contenu principal
def get_main_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('article') or soup.find('div', {'class': 'main-content'})
        return main_content.get_text(strip=True) if main_content else "Contenu principal non trouvé."
    except Exception as e:
        return f"Erreur lors de la récupération de {url}: {e}"

# Fonction pour récupérer les résultats de recherche Google
def get_google_results(query, api_key):
    params = {
        "engine": "google",
        "q": query,
        "num": 10,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "organic_results" in results:
        return [(result["title"], result["link"]) for result in results["organic_results"]]
    else:
        return []

# Interface Streamlit
st.title("Analyse de Similarité Cosinus avec BERT")

# Saisir la requête de recherche
query = st.text_input("Entrez votre requête de recherche:")
api_key = st.text_input("Entrez votre clé API SerpApi:", type="password")

if st.button("Rechercher"):
    if query and api_key:
        results = get_google_results(query, api_key)
        
        # Récupérer le contenu pour chaque URL
        data = []
        for title, url in results:
            content = get_main_content(url)
            data.append({"URL": url, "Contenu": content})

        df = pd.DataFrame(data)

        # Charger le tokenizer et le modèle BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Calculer les embeddings pour chaque contenu
        embeddings = []
        for content in df['Contenu']:
            inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(embedding)

        # Convertir la liste d'embeddings en un tableau numpy
        embeddings_matrix = torch.vstack(embeddings).numpy()

        # Calculer la similarité cosinus
        similarity_matrix = cosine_similarity(embeddings_matrix)

        # Créer un DataFrame pour les scores de similarité
        similarity_df = pd.DataFrame(similarity_matrix, index=df['URL'], columns=df['URL'])

        # Afficher le DataFrame de similarité
        st.write("Scores de Similarité Cosinus:")
        st.dataframe(similarity_df)

        # Optionnel : Sauvegarder le DataFrame de similarité dans un fichier CSV
        similarity_df.to_csv("similarity_scores.csv", index=False)
        st.success("Les scores de similarité ont été sauvegardés dans 'similarity_scores.csv'.")
    else:
        st.error("Veuillez entrer une requête et votre clé API.")