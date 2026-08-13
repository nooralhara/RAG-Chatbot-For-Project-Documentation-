from langchain_classic.document_loaders import TextLoader
from langchain_classic.text_splitter import TokenTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

loader = TextLoader('README(RFC).txt')
document = loader.load()

splitter = TokenTextSplitter(
    chunk_size =600,
    chunk_overlap=75
)
chunks = splitter.split_documents(document)

embeddings = HuggingFaceBgeEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device":"cpu"}
)

vectorDB = FAISS.from_documents(chunks, embeddings)

retriever = vectorDB.as_retriever(
    search_kwargs = {"k": 3}
)