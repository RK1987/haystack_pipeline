from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from nltk.util import ngrams
from collections import Counter

from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline


# Function to calculate Jaccard similarity
def jaccard_similarity(str1, str2):
    ngrams1 = set(ngrams(str1.split(), n=2))
    ngrams2 = set(ngrams(str2.split(), n=2))
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union != 0 else 0

# Function to calculate similarity metrics
def calculate_similarity(predicted, ground_truth):
    # Cosine Similarity
    vectorizer = TfidfVectorizer().fit([predicted, ground_truth])
    tfidf_matrix = vectorizer.transform([predicted, ground_truth])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # Levenshtein Ratio
    levenshtein_sim = levenshtein_ratio(predicted, ground_truth)
    
    # Jaccard Similarity
    jaccard_sim = jaccard_similarity(predicted, ground_truth)
    
    return cosine_sim, levenshtein_sim, jaccard_sim


# Step 1: Initialize an in-memory document store
document_store = InMemoryDocumentStore()

# Step 2: Write the documents into the document store
documents = [
    {"content": "The Eiffel Tower is located in Paris."},
    {"content": "The Great Wall of China is one of the wonders of the world."},
    {"content": "Mount Everest is the highest mountain on Earth."},
    {"content": "The Amazon rainforest is known as the lungs of the Earth."},
    {"content": "Albert Einstein developed the theory of relativity."}
]
document_store.write_documents(documents)

# Step 3: Initialize retriever and reader
retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="facebook/dpr-question_encoder-single-nq-base", passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", use_gpu=False)
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Step 4: Create the pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

# Queries with ground truth
queries = [
    {"query": "Where is the Eiffel Tower?", "ground_truth": "The Eiffel Tower is located in Paris."},
    {"query": "Who developed the theory of relativity?", "ground_truth": "Albert Einstein developed the theory of relativity."},
    {"query": "What is the highest mountain?", "ground_truth": "Mount Everest is the highest mountain on Earth."},
    {"query": "Which forest is called the lungs of the Earth?", "ground_truth": "The Amazon rainforest is known as the lungs of the Earth."},
    {"query": "Which is one of the wonders of the world?", "ground_truth": "The Great Wall of China is one of the wonders of the world."}
]

# Step 5: Evaluate the accuracy
similarity_threshold = 0.8  # Define a similarity threshold for determining correctness
correct_answers_cosine = 0
correct_answers_levenshtein = 0
correct_answers_jaccard = 0

for query_data in queries:
    query = query_data["query"]
    ground_truth = query_data["ground_truth"]
    prediction = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}})
    
    if prediction["answers"]:
        predicted_answer = prediction["answers"][0].answer
        print(f"Query: {query}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted Answer: {predicted_answer}")
        
        # Calculate similarity
        cosine_sim, levenshtein_sim, jaccard_sim = calculate_similarity(predicted_answer, ground_truth)
        print(f"Cosine Similarity: {cosine_sim:.2f}")
        print(f"Levenshtein Ratio: {levenshtein_sim:.2f}")
        print(f"Jaccard Similarity: {jaccard_sim:.2f}")
        print()
        
        # Increment counters if similarity exceeds the threshold
        if cosine_sim >= similarity_threshold:
            correct_answers_cosine += 1
        if levenshtein_sim >= similarity_threshold:
            correct_answers_levenshtein += 1
        if jaccard_sim >= similarity_threshold:
            correct_answers_jaccard += 1
    else:
        print(f"Query: {query}")
        print(f"No answer found!")
        print()

# Calculate and print accuracy for each method
total_queries = len(queries)
cosine_accuracy = correct_answers_cosine / total_queries * 100
levenshtein_accuracy = correct_answers_levenshtein / total_queries * 100
jaccard_accuracy = correct_answers_jaccard / total_queries * 100

print(f"Accuracy (Cosine Similarity): {cosine_accuracy:.2f}%")
print(f"Accuracy (Levenshtein Ratio): {levenshtein_accuracy:.2f}%")
print(f"Accuracy (Jaccard Similarity): {jaccard_accuracy:.2f}%")