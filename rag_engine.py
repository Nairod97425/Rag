import json
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class LocalRAGSystem:
    def __init__(self, persist_directory: str = "db_storage_local"):
        print("🔌 Initialisation des Embeddings (nomic-embed-text)...")
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.prompt_template = None # Stockage du template pour réutilisation
        self.llm = None

    def load_and_index(self, json_path: str):
        """Charge le JSON et indexe les chunks localement"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Fichier introuvable: {json_path}")

        print(f"📂 Chargement des données depuis {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for entry in data:
            base_meta = {
                "source": entry.get("url", "inconnu"), 
                "title": entry.get("title", "Sans titre")
            }
            
            for chunk in entry.get("chunks", []):
                doc = Document(
                    page_content=chunk.get("text", ""),
                    metadata={**base_meta, "chunk_id": chunk.get("id")}
                )
                documents.append(doc)

        print(f"⚙️  Vectorisation de {len(documents)} chunks...")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        print("✅ Indexation terminée !")

    def setup_pipeline(self):
        """Configure la chaîne RAG avec le LLM local"""
        if not self.retriever:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embedding_function
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        print("🧠 Chargement du LLM...")
        self.llm = ChatOllama(model="llama3.2", temperature=0)

        template = """Tu es un assistant médical spécialisé EXCLUSIVEMENT sur le diabète.
        Ta mission est d'aider les utilisateurs uniquement sur ce sujet à partir des documents fournis.

        RÈGLES STRICTES :
        1. 🚫 HORS SUJET : Si la question ne concerne pas le diabète, la glycémie, l'insuline ou la santé liée, refuse poliment de répondre.
           Phrase type : "Je suis un assistant spécialisé uniquement sur le diabète. Je ne peux pas répondre à d'autres sujets."
        
        2. 📄 SOURCES : Utilise UNIQUEMENT le contexte ci-dessous. N'invente rien. Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information".

        3. 💬 LANGUE : Réponds toujours en français.

        Contexte :
        {context}

        Question de l'utilisateur : {question}
        
        Réponse :"""

        self.prompt_template = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([f"[Source: {d.metadata['title']}]\n{d.page_content}" for d in docs])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Méthode simple pour le chat"""
        if not self.chain:
            self.setup_pipeline()
        return self.chain.invoke(question)

    def ask_with_context(self, question: str):
        """Méthode avancée pour Ragas : retourne aussi les sources"""
        if not self.chain:
            self.setup_pipeline()
            
        # 1. Récupération manuelle des documents
        retrieved_docs = self.retriever.invoke(question)
        
        # 2. Formatage pour le LLM
        context_text = "\n\n".join([f"[Source: {d.metadata['title']}]\n{d.page_content}" for d in retrieved_docs])
        
        # 3. Génération de la réponse
        # On recrée une petite chaîne ad-hoc pour avoir le contrôle
        chain = self.prompt_template | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": question})
        
        # 4. Retour formaté pour Ragas
        return {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in retrieved_docs], # Liste de strings brute requise par Ragas
            "source_documents": retrieved_docs
        }