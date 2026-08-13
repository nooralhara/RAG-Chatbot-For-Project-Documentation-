from langchain_classic.document_loaders import TextLoader
from langchain_classic.text_splitter import TokenTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

file_paths = [
    "README(RFC).txt","README(Hate_Speech).txt","README(ResNet50).txt"]
documents = []
while file_paths:
    path = file_paths.pop(0)
    loader = TextLoader(path, encoding="utf-8")
    documents.extend(loader.load())

splitter = TokenTextSplitter(
    chunk_size =700,
    chunk_overlap=125
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceBgeEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device":"cpu"}
)

vectorDB = FAISS.from_documents(chunks, embeddings)

retriever = vectorDB.as_retriever(
    search_kwargs = {"k":5}
)
