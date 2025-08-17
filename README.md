# Information Retrieval System

A three-phase information retrieval system developed for the Information Retrieval course at university during Fall 2024.

## Project Phases

### Phase 1: Structured Information Extraction
- Extracts hierarchical information from PDF documents
- Converts unstructured text to structured JSON format
- Identifies titles, subtitles, and list items based on font sizes and numbering

### Phase 2: Document Retrieval Models
- Implements three retrieval models:
  - Boolean Model (exact keyword matching)
  - Vector Space Model (TF-IDF with cosine similarity)
  - Levenshtein Model (fuzzy string matching)
- Evaluates performance using precision@k and recall@k metrics

### Phase 3: Query Expansion with Thesaurus
- Builds a thesaurus using two approaches:
  - Word co-occurrence statistics
  - Spacy's dependency parsing
- Expands user queries with related terms
- Evaluates impact on retrieval performance

## Usage
Each phase has its own directory with implementation code and documentation. See individual phase READMEs for specific instructions.

## Technologies Used
- Python
- PyPDF2, fitz (PDF processing)
- NLTK (text processing)
- Spacy (NLP)
- scikit-learn (TF-IDF vectorization)
