import pdfplumber
import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def clean_text(text):
    cleaned_text = text.replace("\n", " ").replace("N/A", "")
    return cleaned_text


def process_pdfs_in_directory(directory_path):

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(text)
    return cleaned_text


def clean_text(text):
    # Remove newlines and unwanted text
    cleaned_text = text.replace("\n", " ").strip()  # Replace newlines with spaces
    cleaned_text = cleaned_text.replace("N/A", "").replace(
        ".", ""
    )  # Remove occurrences of 'N/A' and '.'
    return cleaned_text


def load_pdfs_from_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                cleaned_text = clean_text(text)
                texts.append(cleaned_text)
    return texts


def create_documents(texts):
    return [Document(page_content=text) for text in texts]


def main(path):

    persist_directory = "Vectorized_data"
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if os.path.isdir(persist_directory):
        return persist_directory, embeddings_model

    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        texts = load_pdfs_from_directory(path)
        docs = create_documents(texts)

        embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        chunks = text_splitter.split_documents(docs)
        print(chunks)

        persist_directory = "Vectorized_data"
        Chroma.from_documents(
            chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

        return persist_directory, embeddings_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to call module's main function"
    )
    parser.add_argument("path", type=str, help="The path to pass to module")
    args = parser.parse_args()
    main(args.path)
