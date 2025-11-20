import streamlit as st
import time
from rag_engine import LocalRAGSystem
from config import OUTPUT_FILE
import os

# 1. Configuration de la page
st.set_page_config(
    page_title="Assistant Diabète Local",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Assistant Médical IA (Local)")
st.caption("Propulsé par Ollama (Llama 3.2 / Mistral) & RAG")

# 2. Chargement du moteur RAG (mis en cache pour ne pas recharger à chaque clic)
@st.cache_resource
def load_rag_engine():
    """Initialise le RAG une seule fois au démarrage."""
    rag = LocalRAGSystem()
    
    # Vérification automatique de la base de données
    if not os.path.exists(rag.persist_directory):
        with st.spinner("⚙️ Première exécution : Indexation des données..."):
            if os.path.exists(str(OUTPUT_FILE)):
                rag.load_and_index(str(OUTPUT_FILE))
            else:
                st.error(f"Erreur : Le fichier source {OUTPUT_FILE} est introuvable. Lance 'scraper.py' d'abord.")
                st.stop()
    return rag

# On charge le moteur. Le spinner apparaîtra seulement au premier lancement.
try:
    rag_engine = load_rag_engine()
except Exception as e:
    st.error(f"Erreur lors du chargement du moteur RAG : {e}")
    st.stop()

# 3. Gestion de l'historique de chat (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant spécialisé sur le diabète. Posez-moi une question sur vos documents."}
    ]

# 4. Affichage des messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Si le message contient des sources (c'est un message assistant stocké avec sources)
        if "sources" in msg:
            with st.expander("📚 Voir les sources utilisées"):
                for source in msg["sources"]:
                    st.markdown(f"**Source :** {source}")

# 5. Zone de saisie utilisateur
if prompt := st.chat_input("Votre question ici..."):
    # A. Affichage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Génération de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyse des documents en cours..."):
            try:
                # On utilise ta méthode 'ask_with_context' pour récupérer les sources
                result = rag_engine.ask_with_context(prompt)
                answer = result["answer"]
                docs = result["source_documents"]
                
                # Simulation d'effet de frappe (Streaming visuel)
                message_placeholder.markdown(answer)
                
                # Préparation des sources pour l'affichage
                formatted_sources = []
                if docs:
                    with st.expander("📚 Voir les sources utilisées"):
                        for doc in docs:
                            source_title = doc.metadata.get('title', 'Sans titre')
                            source_url = doc.metadata.get('source', '#')
                            st.markdown(f"- **[{source_title}]({source_url})**")
                            st.caption(f"...{doc.page_content[:150]}...") # Aperçu du texte
                            formatted_sources.append(f"[{source_title}]({source_url})")

                # Sauvegarde dans l'historique
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": formatted_sources
                })
                
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")