from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from custom.embeddings.ctranslate2 import Ct2BertEmbeddings
import os

PICKLE_FILE = os.environ['PICKLE_FILE']
EMBEDDINGS_MODEL_NAME = os.environ['EMBEDDINGS_MODEL_NAME']


def ingest_data():
    file_paths = None
    with open('html_files_index.txt', 'r') as file:
        file_paths = file.readlines()

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=20)

    print("Load HTML files locally...")
    for i, file_path in enumerate(file_paths):
        file_path = file_path.rstrip("\n")
        doc = UnstructuredHTMLLoader(file_path).load()
        splits = text_splitter.split_documents(doc)
        docs.extend(splits)
        print(f"{i+1})Split {file_path} into {len(splits)} chunks")

    print("Load data to FAISS store")
    # embeddings = HuggingFaceEmbeddings(
    #    model_name=EMBEDDINGS_MODEL_NAME)
    model_name = ""
    model_kwargs = {'device': 'cpu', 'compute_type': "int8"}
    encode_kwargs = {'batch_size': 32,
                     'convert_to_numpy': True, 'normalize_embeddings': True}
    embeddings = Ct2BertEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    store = FAISS.from_documents(docs, embeddings)

    print(f"Save to {PICKLE_FILE}")
    store.save_local(PICKLE_FILE)
    # with open(PICKLE_FILE, "wb") as f:
    #    pickle.dump(store, f)


if __name__ == "__main__":
    ingest_data()
