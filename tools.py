from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, NotionDBLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import tool
from main import OPENAI_API_KEY


def get_documents(dir: str, types: list):
    documents = []

    for file_type in types:
        loader = DirectoryLoader(path=dir, glob=f"**/*.{file_type}")
        documents.extend(loader.load())

    return documents


def document_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    result = text_splitter.split_documents(documents)

    return result


alloweb_files = ["pdf", "md", "txt", "docx", "pptx", "csv", "xlsx"]


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
documents = get_documents("uploads/", alloweb_files)
ready_documetns = document_splitter(documents)
vectorstore = DocArrayInMemorySearch.from_documents(ready_documetns, embeddings)


@tool
def search_from_documents(text: str):
    """When answering, answer based on the answer provided by this tool"""
    result = vectorstore.similarity_search(text, k=5)
    return result