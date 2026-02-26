import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
CONFIG = {
    "FILE_PATH": "apartments_for_rent.csv",
    "MODEL_NAME": "all-MiniLM-L6-v2",
    "THRESHOLD": 0.80,
    "WEIGHTS": {"hard": 0.5, "semantic": 0.5},
    "USER_PREFS": {
        "query": "A sunny, modern apartment with two bedrooms, in a safe city neighborhood, with a gym and pool, max rent $2500.",
        "max_price": 2500,
        "min_bedrooms": 2,
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PropertyMatchEngine:
    """High-performance Hybrid Search Engine for Real Estate."""

    def __init__(self, config):
        self.config = config
        # Use GPU if available for 10x faster encoding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(config["MODEL_NAME"], device=self.device)
        self.data = pd.DataFrame()
        self.corpus_embeddings = None

    def load_source(self, file_path):
        """Loads data and pre-calculates vectors for instant searching."""
        try:
            logging.info(f"Loading {file_path}...")
            df = pd.read_csv(file_path, encoding="latin-1", sep=";")

            # Standardizing Column Names
            # self.data = df[
            #     ["price", "bedrooms", "bathrooms", "cityname", "body"]
            # ].copy()
            self.data = (
                df[["price", "bedrooms", "bathrooms", "cityname", "body"]]
                .head(1000)
                .copy()
            )
            self.data.columns = [
                "Price",
                "Bedrooms",
                "Bathrooms",
                "City",
                "Description",
            ]

            # Data Cleaning
            self.data["Description"] = self.data["Description"].fillna(
                "No description available"
            )
            self.data["Price"] = pd.to_numeric(self.data["Price"], errors="coerce")
            self.data["Bedrooms"] = (
                pd.to_numeric(self.data["Bedrooms"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            self.data = self.data.dropna(subset=["Price"])

            # THE FIX: Vectorize the entire dataset once
            logging.info(f"Vectorizing {len(self.data)} descriptions. Please wait...")
            descriptions = self.data["Description"].tolist()
            self.corpus_embeddings = self.model.encode(
                descriptions, convert_to_tensor=True, show_progress_bar=True
            )

            logging.info("Engine ready. Matrix Math initialized.")
            return self
        except Exception as e:
            logging.error(f"Critical Failure in load_source: {e}")
            return self

    def _calculate_hard_scores(self, prefs):
        """Vectorized hard filter calculation using Pandas logic."""
        price_match = (self.data["Price"] <= prefs["max_price"]).astype(float)
        bed_match = (self.data["Bedrooms"] >= prefs["min_bedrooms"]).astype(float)
        return (price_match + bed_match) / 2

    def run_search(self, prefs):
        """Executes hybrid search in milliseconds using Matrix Math."""
        if self.data.empty or self.corpus_embeddings is None:
            logging.error("Search failed: No data loaded.")
            return pd.DataFrame()

        # 1. Encode query (Single AI call)
        query_emb = self.model.encode(prefs["query"], convert_to_tensor=True)

        # 2. Semantic calculation (Instant Matrix Multiplication)
        logging.info("Computing semantic similarity...")
        semantic_scores = (
            util.cos_sim(query_emb, self.corpus_embeddings).flatten().tolist()
        )

        # 3. Combine results
        results_df = self.data.copy()
        results_df["Semantic_Score"] = semantic_scores
        results_df["Hard_Score"] = self._calculate_hard_scores(prefs)

        # 4. Apply Weights
        w_h = self.config["WEIGHTS"]["hard"]
        w_s = self.config["WEIGHTS"]["semantic"]
        results_df["Final_Score"] = (w_h * results_df["Hard_Score"]) + (
            w_s * results_df["Semantic_Score"]
        )

        return results_df.sort_values(by="Final_Score", ascending=False)


# --- EXECUTION ---
if __name__ == "__main__":
    engine = PropertyMatchEngine(CONFIG)
    engine.load_source(CONFIG["FILE_PATH"])

    print("\n" + "=" * 50)
    print("🚀 HYBRID SEARCH ENGINE READY")
    print("=" * 50)

    while True:
        print("\n" + "-" * 30)
        query = input("🔍 Description (or 'exit'): ")
        if query.lower() == "exit":
            break

        # Pragmatic: Ask for the hard filters too!
        try:
            max_p = int(input("💰 Max Price (e.g. 3000): ") or 2500)
            min_b = int(input("🛏️ Min Bedrooms (e.g. 2): ") or 2)
        except ValueError:
            print("Invalid input, using defaults.")
            max_p, min_b = 2500, 2

        current_prefs = {"query": query, "max_price": max_p, "min_bedrooms": min_b}

        final_ranked = engine.run_search(current_prefs)

        # Let's see the Top 5 regardless of threshold to see what's happening
        print("\n--- Top 5 Closest Matches Found ---")
        print(
            final_ranked[
                [
                    "Price",
                    "Bedrooms",
                    "City",
                    "Final_Score",
                    "Hard_Score",
                    "Semantic_Score",
                ]
            ]
            .head(5)
            .to_markdown(index=False)
        )
