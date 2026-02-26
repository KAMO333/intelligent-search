# Intelligent Search Prototype

This repository contains a simple hybrid search prototype implemented in Python. It combines structured filtering (hard filters) with semantic similarity scoring to rank apartment rental listings.

## Overview

The main script, `hybrid_search_prototype.py`, performs the following steps:

1. **Data Loading and Cleaning**
   - Reads `apartments_for_rent.csv` using pandas with latin-1 encoding.
   - Selects relevant columns, renames them, and cleans missing data.
   - Ensures numeric types for `Price` and `Bedrooms` (using Pandas `Int64` dtype).
   - Drops rows with invalid prices and short descriptions, limiting to the first 1000 records for the prototype.

2. **Hybrid Score Calculation**
   - Calculates a hard filter score based on maximum price and required bedrooms.
   - Computes semantic similarity between the user query and listing descriptions using a sentence transformer (`all-MiniLM-L6-v2`).
   - Combines the hard and semantic scores with configurable weights.

3. **Results Output**
   - Sorts listings by the final score.
   - Applies a relevance threshold to display only top matches.
   - Prints debug tables of the top results.

## Configuration

Key variables defined at the top of the script:

```python
FILE_PATH = 'apartments_for_rent.csv'
USER_QUERY = "A sunny, modern apartment with two bedrooms, in a safe city neighborhood, with a gym and pool, max rent $2,500."
THRESHOLD = 0.80  # Relevance threshold for filtering results
WEIGHT_HARD = 0.5
WEIGHT_SEMANTIC = 0.5
```

The script uses a simulated maximum price and required bedrooms when invoked as a standalone program.

## Prerequisites

- Python 3.7+
- `pandas` and `numpy` for data handling
- `sentence-transformers` for computing semantic embeddings

Install dependencies with:

```bash
pip install pandas numpy sentence-transformers
```

## Usage

Run the prototype from the command line:

```bash
python hybrid_search_prototype.py
```

It will load and clean the data, initialize the semantic model, calculate scores, and print the top ranked listings along with debug information.

You can modify the query, threshold, weights, or add real user inputs as needed.

## Notes

- The dataset `apartments_for_rent.csv` is expected to have semicolon-separated fields with columns such as `price`, `bedrooms`, `cityname`, and `body` for descriptions.
- The script currently limits processing to the first 1000 cleaned listings for performance during prototyping.
- The hybrid scoring mechanism is easily extensible to include additional hard filters or change weighting.

## License

This is a prototype for demonstration purposes; adapt and extend under your preferred license.
