# RAG Chatbot for ML Project Documentation

A Retrieval-Augmented Generation (RAG) chatbot that answers question about my three machine learning projects:
Hate-Speech Detection, Teeth Disease Classification, and Stroke Prediction. Built with FAISS for vector search,
Sentence-Transformers for embeddings, and an LLM via OpenRouter, deployed as a streamlit web application.


## Overview

The chatbot retrieves relevant chunks from the three project READMEs based on user inquiries, then feeds that context into the LLM to generate answers.

1. The three projects are loaded as .txt files and split into chunks.
2. Each chunk is embedded into a vector using Sentence-Transformers.
3. Embeddings are then indexed in FAISS for fast similarity search.
4. For each inquiry, the top-k most relevant chunks are retrieved.
5. Retrieved chunks and the inquiry are inserted into a prompt template.
6. The prompt is sent to the LLM and the response is shown on the streamlit interface.


## Structure

1. app.py #Streamlit web application
2. RAG.py #Builds the RAG Pipeline
3. LLM_MODEL.py #Sets up the LLM connection with OpenRouter
4. prompts.py # Prompt template
5. README(Hate_Speech).txt + README(ResNet50).txt + README(RFC).txt #Source Documents
6. requirements.txt #Python Dependencies

## Tech Stack

1. **Streamlit** : Chat UI + Deployment
2. **Langchain** : Document Loading + Splitting + Prompt Template
3. **FAISS** : Vector Similarity Search
4. **Sentence-Transformers** : Text Embedding
5. **OpenRouter** : LLM API Access

## Notes

The knowledge base is limited to the projects' READMEs mentioned above, the chatbot only answers based on context in those files

## Made By: NoorAldeen Faruq AlHara
