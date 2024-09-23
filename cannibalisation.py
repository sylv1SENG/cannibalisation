import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

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
st.title("Analyse de Similarité Cosinus avec Universal Sentence Encoder (USE)")

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
                # Charger Universal Sentence Encoder (USE) depuis TensorFlow Hub
                try:
                    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
                    model = hub.load(model_url)
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle USE: {e}")
                    st.stop()

                # Calculer les embeddings pour chaque contenu
                embeddings = model(df['Contenu'].tolist())

                # Calculer la similarité cosinus entre les embeddings
                similarity_matrix = tf.keras.backend.eval(tf.matmul(embeddings, embeddings, transpose_b=True))

                # Créer une liste pour stocker les résultats de similarité sous forme de paires
                similarity_data = []
                for i, url_1 in enumerate(df['URL']):
                    for j, url_2 in enumerate(df['URL']):
                        if i != j:  # On évite de comparer une URL avec elle-même
                            similarity_data.append({
                                "URL": url_1,
                                "URL Comparée": url_2,
                                "Score de Similarité": similarity_matrix[i][j]
                            })

                # Créer un DataFrame pour les scores de similarité
                similarity_df = pd.DataFrame(similarity_data)

                # Afficher le DataFrame de similarité
                st.write("Scores de Similarité Cosinus (paires d'URLs):")
                st.dataframe(similarity_df)

                # Sauvegarder le DataFrame de similarité dans un fichier CSV
                csv = similarity_df.to_csv(index=False)
                st.download_button("Télécharger les scores de similarité", csv, "similarity_scores.csv")
    else:
        st.error("Veuillez uploader un fichier Excel avec les URLs.")
