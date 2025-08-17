import os
import nltk
import json
import PyPDF2
import Levenshtein as lev
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

def extract_text_from_pdfs(path):
    documents = []
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            with open(os.path.join(path, filename), 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                documents.append(text)
                filenames.append(filename)
    return documents, filenames


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = " ".join([word for word in tokens if word.lower() not in stop_words])
    return filtered_text


def apply_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_text = " ".join([stemmer.stem(word) for word in tokens])
    return stemmed_text


def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in tokens])
    return lemmatized_text


class BooleanModel:
    def __init__(self, documents):
        self.documents = documents
        self.indexed_docs = self.index_documents(documents)

    def index_documents(self, documents):
        index = {}
        for doc_id, doc in enumerate(documents):
            for word in set(doc.split()):
                word = word.lower()
                if word not in index:
                    index[word] = []
                index[word].append(doc_id)
        return index

    def process_query(self, query):
        query_tokens = query.split()
        terms = []
        operators = []
        temp_tokens = []

        for token in query_tokens:
            if token.lower() in ['and', 'or', 'not']:
                operators.append(token.lower())
                terms.append(' '.join(temp_tokens))
                temp_tokens = []
            else:
                temp_tokens.append(token)
        terms.append(' '.join(temp_tokens))

        result_set = None
        for i, term in enumerate(terms):
            term = term.lower().strip()
            if term:
                docs_for_term = set(self.indexed_docs.get(term, []))

                if result_set is None:
                    result_set = docs_for_term
                else:
                    if operators[i-1] == 'and':
                        result_set &= docs_for_term
                    elif operators[i-1] == 'or':
                        result_set |= docs_for_term
                    elif operators[i-1] == 'not':
                        result_set -= docs_for_term

        return result_set

    def search(self, query):
        result_docs = self.process_query(query)
        return list(result_docs) if result_docs else []


class VectorSpaceModel:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def rank_documents(self, query):
        query_tfidf = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        return cosine_similarities


class LevenshteinModel:
    def __init__(self, documents):
        self.documents = documents

    def rank_documents(self, query):
        query_terms = query.split()
        scores = []
        for doc in self.documents:
            score = 0
            for term in query_terms:
                normalized_distances = [
                    1 - (lev.distance(term, word) / max(len(term), len(word)))
                    for word in doc.split()
                ]
                score += max(normalized_distances) if normalized_distances else 0
            scores.append(score / len(query_terms) if query_terms else 0)
        return scores


def calculate_precision_recall(retrieved_docs, relevant_docs):
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    true_positives = len(retrieved_set & relevant_set)
    print(f"True positives: {true_positives}, Retrieved: {len(retrieved_set)}, Relevant: {len(relevant_set)}")
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0

    return precision, recall


def calculate_p_at_k(retrieved_docs, relevant_docs, k):
    retrieved_set = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    true_positives = len([doc for doc in retrieved_set if doc in relevant_set])
    precision_at_k = true_positives / k if k > 0 else 0

    return precision_at_k


def calculate_r_at_k(retrieved_docs, relevant_docs, k):
    retrieved_set = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    true_positives = len([doc for doc in retrieved_set if doc in relevant_set])
    recall_at_k = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0

    return recall_at_k


def plot_vectors(scores, filenames, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(filenames[:10], scores[:10], color='skyblue')

    ax.set_xlabel('Score')
    ax.set_ylabel('Document')
    ax.set_title(f'{model_name} - Top 10 Document Scores')
    plt.grid(True)
    plt.show()


def search(query, documents, filenames, relevant_docs, k, model_type='vector'):
    if model_type == 'boolean':
        boolean_model = BooleanModel(documents)
        doc_ids = boolean_model.search(query)

        retrieved_docs = list(set(filenames[i] for i in doc_ids))

        precision, recall = calculate_precision_recall(retrieved_docs, relevant_docs)

        results = [(1 if filenames[i] in retrieved_docs else 0, filenames[i]) for i in range(len(filenames))]
        results = [result for result in results if result[0] == 1]

        return results, precision, recall

    elif model_type == 'vector':
        vector_model = VectorSpaceModel(documents)
        scores = vector_model.rank_documents(query)

        ranked_docs = sorted([(score, filenames[i]) for i, score in enumerate(scores) if score > 0], reverse=True)
        ranked_docs = list(dict.fromkeys(ranked_docs))

        retrieved_docs = [doc for _, doc in ranked_docs]
        p_at_k = calculate_p_at_k(retrieved_docs, relevant_docs, k)
        r_at_k = calculate_r_at_k(retrieved_docs, relevant_docs, k)

        if ranked_docs:
            plot_vectors([score for score, _ in ranked_docs], retrieved_docs, 'Vector Space Model (TF-IDF)')

        return ranked_docs, p_at_k, r_at_k

    elif model_type == 'levenshtein':
        lev_model = LevenshteinModel(documents)
        scores = lev_model.rank_documents(query)

        ranked_docs = [(score, filenames[i]) for i, score in enumerate(scores) if score > 0]
        ranked_docs = sorted(ranked_docs, reverse=True)
        ranked_docs = list(dict.fromkeys(ranked_docs))

        retrieved_docs = [doc for _, doc in ranked_docs]
        p_at_k = calculate_p_at_k(retrieved_docs, relevant_docs, k)
        r_at_k = calculate_r_at_k(retrieved_docs, relevant_docs, k)

        if ranked_docs:
            plot_vectors([score for score, _ in ranked_docs], retrieved_docs, 'Levenshtein Model')

        return ranked_docs, p_at_k, r_at_k


if __name__ == "__main__":

    json_path = r"E:\university\semester-9\information retrieval\project\1\1-2\updated-queries.json"

    with open(json_path, 'r', encoding='utf-8') as file:
        query_data = json.load(file)

    k = int(input("Enter the value of k: "))

    documents_path = r"E:\university\semester-9\information retrieval\project\1\1-2\Documents"
    documents, filenames = extract_text_from_pdfs(documents_path)

    preprocessing_modes = [
        ('No Preprocessing', lambda x: x),
        ('Remove Stopwords', remove_stopwords),
        ('Stemming', apply_stemming),
        ('Lemmatization', apply_lemmatization),
    ]

    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break

        matched_entry = next((entry for entry in query_data if entry['query_text'] == query_text), None)

        if not matched_entry:
            print("Query not found in the benchmark JSON.")
            continue

        relevant_docs = [resp['doc_name'] for resp in matched_entry['responses']]

        print("\nSelect a model to run:")
        print("1. Boolean Model")
        print("2. Vector Space Model (TF-IDF)")
        print("3. Levenshtein Model")
        model_choice = input("Enter the number of the model: ")

        if model_choice not in ['1', '2', '3']:
            print("Invalid model choice. Please choose 1, 2, or 3.")
            continue

        for mode_name, preprocess in preprocessing_modes:
            print(f"\nRunning models with {mode_name}...")

            preprocessed_documents = [preprocess(doc) for doc in documents]

            if model_choice == '1':
                print("\nRunning Boolean Model...")
                boolean_results, boolean_p_at_k, boolean_r_at_k = search(query_text, preprocessed_documents, filenames,
                                                                         relevant_docs, k, model_type='boolean')
                for result in boolean_results:
                    print(f"Document: {result[1]}, Status: {result[0]}")
                print(f"Boolean Model Precision: {boolean_p_at_k:.4f}, Recall: {boolean_r_at_k:.4f}")

            elif model_choice == '2':
                print("\nRunning Vector Space Model (TF-IDF)...")
                vector_results, vector_p_at_k, vector_r_at_k = search(query_text, preprocessed_documents, filenames,
                                                                      relevant_docs, k, model_type='vector')
                for score, filename in vector_results:
                    print(f"Document: {filename}, Cosine Similarity: {score:.4f}")
                print(f"Vector Space Model P@{k}: {vector_p_at_k:.4f}, R@{k}: {vector_r_at_k:.4f}")

            elif model_choice == '3':
                print("\nRunning Levenshtein Model...")
                lev_results, lev_p_at_k, lev_r_at_k = search(query_text, preprocessed_documents, filenames, relevant_docs,
                                                             k, model_type='levenshtein')
                for score, filename in lev_results:
                    print(f"Document: {filename}, Score: {score}")
                print(f"Levenshtein Model P@{k}: {lev_p_at_k:.4f}, R@{k}: {lev_r_at_k:.4f}")

