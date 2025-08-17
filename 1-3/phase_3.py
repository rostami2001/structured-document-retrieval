import os
import nltk
import json
import PyPDF2
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Load pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


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


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def build_thesaurus(documents):
    tokenized_docs = [preprocess_text(doc) for doc in documents]
    word2vec_model = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)

    thesaurus = {}
    for word in word2vec_model.wv.index_to_key:
        similar_words = word2vec_model.wv.most_similar(word, topn=5)
        thesaurus[word] = [w for w, _ in similar_words]
    return thesaurus


def expand_query(query, thesaurus, weight=0.5):
    tokens = word_tokenize(query)
    expanded_query = {token: 1.0 for token in tokens}
    for token in tokens:
        if token in thesaurus:
            for related_word in thesaurus[token]:
                expanded_query[related_word] = weight
    return expanded_query


class VectorSpaceModel:
    def __init__(self, documents):
        self.document_embeddings = sbert_model.encode(documents)
        # Ensure embeddings are 2D
        if len(self.document_embeddings.shape) == 3:
            self.document_embeddings = self.document_embeddings.squeeze(axis=-1)

    def rank_documents(self, query_embedding):
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        cosine_similarities = cosine_similarity(query_embedding, self.document_embeddings)
        return cosine_similarities.flatten()



def calculate_metrics(retrieved_docs, relevant_docs, k=15):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    true_positives = len(retrieved_set & relevant_set)
    precision_at_k = true_positives / len(retrieved_set) if retrieved_set else 0
    recall_at_k = true_positives / len(relevant_set) if relevant_set else 0
    f1_score_at_k = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)) if (
                                                                                                       precision_at_k + recall_at_k) > 0 else 0

    return precision_at_k, recall_at_k, f1_score_at_k


if __name__ == "__main__":
    documents_path = r"E:\university\semester-9\information retrieval\project\1-2\Documents"
    query = input("Enter your query: ")

    documents, filenames = extract_text_from_pdfs(documents_path)
    preprocessed_documents = [" ".join(preprocess_text(doc)) for doc in documents]

    thesaurus = build_thesaurus(preprocessed_documents)

    expanded_query = expand_query(query, thesaurus)
    print(f"Original Query: {query}")
    print(f"Expanded Query Terms: {expanded_query}")

    # Combine query terms into a single embedding with weights
    query_embedding = sum(sbert_model.encode([term]) * weight for term, weight in expanded_query.items())

    vector_model = VectorSpaceModel(preprocessed_documents)
    scores = vector_model.rank_documents(query_embedding)

    ranked_docs = [(score, filenames[i]) for i, score in enumerate(scores) if score > 0]
    ranked_docs.sort(reverse=True, key=lambda x: x[0])

    retrieved_docs = [doc for _, doc in ranked_docs]
    print(f"\nRetrieved Documents: {len(retrieved_docs)}")
    for score, filename in ranked_docs[:10]:
        print(f"Document: {filename}, Score: {score:.4f}")

    # Load relevant documents from JSON
    json_path = r"E:\university\semester-9\information retrieval\project\1-2\updated-queries.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    query_id = int(input("Enter query_id to compare results: "))
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