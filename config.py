from pathlib import Path

# === URLs à scraper ===
URLS_TO_SCRAPE = [
    "https://www.ameli.fr/la-reunion/assure/sante/themes/diabete-adulte/diabete-symptomes-evolution/diagnostic-diabete",
    "https://www.federationdesdiabetiques.org/information/diabete/chiffres-france",
    "https://www.has-sante.fr/jcms/p_3228739/fr/diabete-de-type-2",
    "https://www.santepubliquefrance.fr/maladies-et-traumatismes/diabete/la-maladie/#tabs",
    "https://www.santepubliquefrance.fr/maladies-et-traumatismes/diabete/notre-action/#tabs",
    "https://www.santepubliquefrance.fr/maladies-et-traumatismes/diabete/donnees/#tabs",
    # Ajoute-en autant que tu veux
]

# === Configuration du scraping ===
SCRAPING_CONFIG = {
    "threads": 4,              # Augmente si tu as une bonne connexion
    "sleep_time": 2,           # Respectueux mais plus rapide (trafilatura gère déjà le rate-limiting)
    "chunk_size": 1024,        # Taille idéale pour la plupart des modèles (surtout avec overlap)
    "overlap": 200,            # 20% d'overlap → très bon pour la cohérence contextuelle
    "max_retries": 3,          # Nouvelle option
    "timeout": 30,
}

# === Chemins de sortie ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "scraped_data.json"
INDIVIDUAL_OUTPUT = True   # Sauvegarder aussi un fichier par URL ?