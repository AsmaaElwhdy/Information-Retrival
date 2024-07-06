import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pyarabic.araby as araby
import stanza

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Arabic NLP pipeline with necessary processors
nlp = stanza.Pipeline(lang='ar', processors='tokenize,lemma')



# Cache for preprocessed documents
preprocessed_documents_cache = {}


# Function to preprocess a single document
def preprocess_doc(doc):
    """ Preprocess a single document. """
    # Remove Tashkeel and Punctuation
    doc = re.sub(r'ٱ', 'ا', doc)
    doc = re.sub(r'[^\w\s]', '', doc)
    # Arabic tokenization
    tokens_before_nlp = araby.tokenize(doc)
    # Stop-word removal
    arabic_stop_words = set(stopwords.words('arabic'))
    filtered_tokens = [token for token in tokens_before_nlp if token not in arabic_stop_words]
    # Process the document with Stanza
    doc = nlp(' '.join(filtered_tokens))
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    # Join tokens back into text
    preprocessed_doc = ' '.join(lemmas)
    # Remove Tashkeel
    preprocessed_doc = araby.strip_tashkeel(preprocessed_doc)
    return preprocessed_doc,filtered_tokens

tokens_doc ={}
doc_tokens ={}
# Function to preprocess all documents
def preprocess_all_documents(documents):
    """ Preprocess all documents and cache them. """
    for index, doc in tqdm(enumerate(documents), total=len(documents)):
        preprocessed_documents_cache[index],tokens_doc[index] = preprocess_doc(doc)
    for index, tokens in tqdm(tokens_doc.items()):
        doc_tokens[data['poem_title'][index]] = tokens
        
def create_inverted_index(query_tokens):
    inverted_index = {}
    documents = []
    for token in query_tokens:
        inverted_index[token] = []  
    for poem, poem_tokens in tqdm(doc_tokens.items()):
        for token in query_tokens :
            if token in poem_tokens :
                inverted_index[token].append(poem)  
                
    return inverted_index
# Function to perform a search
def search(query, documents):
    """ Perform a search by comparing the query with the preprocessed documents. """
    preprocessed_query,lemmas_query = preprocess_doc(query)
    terms_tokens = list(set(lemmas_query))
    # Ensure all documents are preprocessed and cached
    if len(preprocessed_documents_cache) != len(documents):
        preprocess_all_documents(documents)
    
    # Retrieve preprocessed documents from cache
    preprocessed_documents = [preprocessed_documents_cache[idx] for idx in range(len(documents))]
    
    # Calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents + [preprocessed_query])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Sort documents by similarity
    ranked_indices = similarities.argsort()[0][::-1]
    ranked_documents = [(idx, documents[idx], similarities[0][idx]) for idx in ranked_indices]
    
    return ranked_documents,terms_tokens

# Streamlit interface

st.title("Search Engine")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)[:50]
    st.write("Original Data:", data.head())

    text_column = st.selectbox("Select the text column:", options=data.columns)
    query = st.text_input("Enter your query:")
    
    if st.button("Search"):
        if query:
            if text_column is None:
                st.error("Please select a text column before searching.")
            else:
                documents = list(data[text_column])
                
                 
                search_results,terms_tokens = search(query, documents)
                st.write("Search Results:")
                inverted_index = create_inverted_index(terms_tokens)
                st.write(inverted_index)
                top_five_results = []
                for idx, doc, similarity in search_results[:5]:
                    preprocessed_doc = preprocessed_documents_cache[idx]
                    top_five_results.append({"Poem": data['poem_title'][idx], "Similarity": similarity, "Original Document": doc})

                # Display top ten search results in DataFrame
                df_top_five_results = pd.DataFrame(top_five_results)
                st.dataframe(df_top_five_results)

                # Button to download DataFrame as CSV
                csv = df_top_five_results.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name="top_ten_search_results.csv")

