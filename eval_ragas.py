import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag_engine import LocalRAGSystem

# ==========================================
# CONFIGURATION DU "JUGE" (IA LOCALE)
# ==========================================
print("⚖️  Configuration du Juge Ragas...")

JUDGE_MODEL_NAME = "mistral"

print(f"   Utilisation du modèle : {JUDGE_MODEL_NAME}")
judge_llm = ChatOllama(model=JUDGE_MODEL_NAME, temperature=0, format="json")
judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ==========================================
# DATASET DE TEST
# ==========================================
TEST_QUESTIONS = [
    "Quels sont les symptômes principaux du diabète ?",
    "Comment diagnostique-t-on un diabète de type 2 ?",
    # "Quels sont les chiffres du diabète en France ?",
    # "Quelle est la différence entre diabète type 1 et type 2 ?" 
]

# CORRECTION ICI : Ce sont des Strings simples, pas des listes ["..."]
GROUND_TRUTHS = [
    "Soif intense, urines abondantes, fatigue, perte de poids.",
    "Prise de sang à jeun (glycémie > 1,26 g/l à deux reprises).",
    # "Plus de 3,5 millions de personnes traitées en 2020.",
    # "Le type 1 est auto-immun (insuline), le type 2 est lié au mode de vie et à l'âge."
]

def build_dataset():
    """Pose les questions au RAG et construit le dataset"""
    rag = LocalRAGSystem()
    
    if not os.path.exists(rag.persist_directory):
        print("⚠️ Base de données introuvable. Lance d'abord 'main.py' puis 'main_local.py'.")
        return None

    print(f"🤖 Interrogation du RAG sur {len(TEST_QUESTIONS)} questions...")
    
    # CORRECTION ICI : Utilisation des noms de colonnes Ragas v0.2 officiels
    data = {
        "user_input": [],        # Au lieu de "question"
        "response": [],          # Au lieu de "answer"
        "retrieved_contexts": [],# Au lieu de "contexts"
        "reference": []          # Au lieu de "ground_truth"
    }

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"   [{i+1}/{len(TEST_QUESTIONS)}] Question : {q}")
        try:
            result = rag.ask_with_context(q)
            
            data["user_input"].append(result["question"])
            data["response"].append(result["answer"])
            data["retrieved_contexts"].append(result["contexts"])
            data["reference"].append(GROUND_TRUTHS[i]) # Ajout de la string directe
        except Exception as e:
            print(f"❌ Erreur sur la question '{q}': {e}")

    return Dataset.from_dict(data)

def run_evaluation():
    dataset = build_dataset()
    if not dataset:
        return

    print("\n📊 Lancement de l'évaluation (Patience, c'est lent en local)...")
    
    # On configure pour éviter que ça plante si c'est trop long
    my_run_config = RunConfig(
        timeout=300,      # On laisse 5 minutes par question (large sécurité)
        max_retries=1,    # On réessaie 1 fois en cas d'échec
        max_workers=1     # <--- LE SECRET : Une seule évaluation à la fois !
    )
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=my_run_config # <--- Ajout de la config
    )

    print("\n🏆 RÉSULTATS :")
    print(results)

    # Sauvegarde
    df = results.to_pandas()
    df.to_csv("resultats_evaluation.csv", index=False)
    print("\n💾 Résultats sauvegardés dans 'resultats_evaluation.csv'")

if __name__ == "__main__":
    run_evaluation()