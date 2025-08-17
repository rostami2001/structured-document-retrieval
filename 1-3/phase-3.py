import os
import nltk
import json
import PyPDF2
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('punkt')

def extract_text_from_pdfs(path):
    documents, filenames = [], []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            with open(os.path.join(path, filename), 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''.join(page.extract_text() for page in pdf_reader.pages)
                documents.append(text)
                filenames.append(filename)
    return documents, filenames

def preprocess_text(text, mode="lemmatization"):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]

    if mode == "stemming":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif mode == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def build_thesaurus(documents):
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for doc in documents:
        tokens = word_tokenize(doc)
        for i, word in enumerate(tokens):
            for j in range(max(0, i - 2), min(len(tokens), i + 3)):
                if i != j:
                    co_occurrence[word][tokens[j]] += 1

    thesaurus = {}
    for word, neighbors in co_occurrence.items():
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        thesaurus[word] = [neighbor for neighbor, _ in sorted_neighbors[:5]]

    return thesaurus

def expand_query(query, thesaurus):
    tokens = word_tokenize(query)
    expanded_query = set(tokens)
    for token in tokens:
        if token in thesaurus:
            expanded_query.update(thesaurus[token])
    return " ".join(expanded_query)

class VectorSpaceModel:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def rank_documents(self, query):
        query_tfidf = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        return cosine_similarities

def calculate_metrics(retrieved_docs, relevant_docs, k=15):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)

    precision_at_k = true_positives / len(retrieved_set) if retrieved_set else 0

    recall_at_k = true_positives / len(relevant_set) if relevant_set else 0

    f1_score_at_k = (2 * precision_at_k * recall_at_k / (
                precision_at_k + recall_at_k)) if precision_at_k + recall_at_k > 0 else 0

    return precision_at_k, recall_at_k, f1_score_at_k

if __name__ == "__main__":
    documents_path = r"E:\university\semester-9\information retrieval\project\1-2\Documents"

    query = input("Enter your query: ")
    preprocessing_mode = "lemmatization"

    documents, filenames = extract_text_from_pdfs(documents_path)
    preprocessed_documents = [preprocess_text(doc, mode=preprocessing_mode) for doc in documents]

    thesaurus = build_thesaurus(preprocessed_documents)

    expanded_query = expand_query(query, thesaurus)
    print(f"Original Query: {query}")
    print(f"Expanded Query: {expanded_query}")

    vector_model = VectorSpaceModel(preprocessed_documents)
    scores = vector_model.rank_documents(expanded_query)

    ranked_docs = [(score, filenames[i]) for i, score in enumerate(scores) if score > 0]
    retrieved_docs = [doc for _, doc in ranked_docs]
    print(f"Total Documents Retrieved: {len(retrieved_docs)}")
    print("\nRetrieved Documents:", retrieved_docs)

    print(f"Total Documents Retrieved: {len(retrieved_docs)}")
    for score, filename in ranked_docs:
        print(f"Document: {filename}, Cosine Similarity: {score:.4f}")

    print("\nTop Ranked Documents:")
    for score, filename in ranked_docs[:10]:
        print(f"Document: {filename}, Score: {score:.4f}")

    json_path = r"E:\university\semester-9\information retrieval\project\1-2\updated-queries.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    query_id = int(
        input("Enter query_id to compare results: "))

    relevant_docs = [response['doc_name'] for response in data[query_id - 1]['responses']]

    precision_at_k, recall_at_k, f1_score_at_k = calculate_metrics(retrieved_docs, relevant_docs, k=15)
    print(f"\nMetrics for Query ID {query_id}:")
    print(f"P@k: {precision_at_k:.4f}, R@k: {recall_at_k:.4f}, F1-Score@k: {f1_score_at_k:.4f}")

    if ranked_docs:
        plt.figure(figsize=(10, 6))
        plt.barh([doc for _, doc in ranked_docs[:10]], [score for score, _ in ranked_docs[:10]], color='skyblue')
        plt.xlabel('Score')
        plt.ylabel('Document')
        plt.title('Top 10 Documents After Query Expansion')
        plt.grid(True)
        plt.show()
