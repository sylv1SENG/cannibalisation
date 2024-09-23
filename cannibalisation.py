import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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

# Interface Streamlit
st.title("Analyse de Similarité Cosinus avec BERT")

# Uploader le fichier Excel
uploaded_file = st.file_uploader("Choisissez un fichier Excel avec les URLs", type=["xls", "xlsx"])

if st.button("Analyser"):
    if uploaded_file is not None:
        # Lire le fichier Excel
        df_urls = pd.read_excel(uploaded_file)
        
        if "Address" not in df_urls.columns:
            st.error("Le fichier Excel doit contenir une colonne nommée 'Address'.")
        else:
            urls = df_urls["Address"].tolist()

            # Créer une liste pour stocker les résultats
            data = []

            # Récupérer le contenu pour chaque URL
            for url in urls:
                content = get_main_content(url)
                data.append({"URL": url, "Contenu": content})

            df = pd.DataFrame(data)

            # Filtrer les lignes où le contenu est non vide
            df = df[df['Contenu'] != ""]
            df = df[df['Contenu'] != "Contenu principal non trouvé."]
            df = df.dropna(subset=['Contenu'])

            if df.empty:
                st.warning("Aucun contenu valide n'a été trouvé pour les URLs fournies.")
            else:
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
                embeddings_matrix = torch.vstack([torch.tensor(emb) for emb in embeddings]).numpy()

                # Calculer la similarité cosinus
                similarity_matrix = cosine_similarity(embeddings_matrix)

                # Créer un DataFrame pour les scores de similarité avec les URLs filtrées
                similarity_df = pd.DataFrame(similarity_matrix, index=df['URL'], columns=df['URL'])

                # Afficher le DataFrame de similarité
                st.write("Scores de Similarité Cosinus:")
                st.dataframe(similarity_df)

                # Optionnel : Sauvegarder le DataFrame de similarité dans un fichier CSV
                csv = similarity_df.to_csv(index=True)
                st.download_button("Télécharger les scores de similarité", csv, "similarity_scores.csv")
    else:
        st.error("Veuillez uploader un fichier Excel avec les URLs.")
