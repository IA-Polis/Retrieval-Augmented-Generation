import pdfplumber, os, argparse, sys, PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd
#from natsort import natsorted
#from time import sleep
import shutil

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
    # Remove newlines and unwanted text
    cleaned_text = text.replace("\n", " ").strip()  # Replace newlines with spaces
    cleaned_text = cleaned_text.replace("N/A", "").replace("_", "")  # Remove occurrences of 'N/A' and '.'
    return cleaned_text


def load_pdf(file_path):
    texts = []
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        cleaned_text = clean_text(text)
        texts.append(cleaned_text)
    return texts


def create_documents(texts):
    return [Document(page_content=text) for text in texts]


def main():

    df = pd.DataFrame(columns=["Document", "item 6", "item7"])

    count = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2900, 

        chunk_overlap=2100,

        length_function=len,

        add_start_index=True
    )

    path = sys.argv[1]
    order_path = sys.argv[2]
    files = os.listdir(path)
    total = 0
    false = 0
    primeiro = 0
    segundo = 0
    
    order = pd.read_excel(order_path,  sheet_name="Bulas", usecols=['Nome do arquivo da bula'])
    order = order.dropna(subset=['Nome do arquivo da bula'])
    order['Nome do arquivo da bula'] = order['Nome do arquivo da bula'].apply(lambda x: x[:-4])
    order = list(order['Nome do arquivo da bula'])

    for filename in order:
            total += 1
            file_path = os.path.join(path, str(f"{filename}.pdf"))
            print(file_path) 

            texts = load_pdf(file_path)
            doc = create_documents(texts)
            chunks = text_splitter.split_documents(doc)

            embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
            chunks = text_splitter.split_documents(doc)

            persist_directory = f"itens/{filename}"
            Chroma.from_documents(
                chunks,
                embedding=embeddings_model,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
                    
            query1 = "6. COMO DEVO USAR ESTE MEDICAMENTO?"
            query2 = "7. O QUE DEVO FAZER QUANDO EU ME ESQUECER DE USAR ESTE MEDICAMENTO?"

            vectorized_db = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings_model
            )

            result_q1 = vectorized_db.similarity_search(query1, k=2)
            result_q2 = vectorized_db.similarity_search(query2, k=2)
            print(result_q1)
            exit()

            query1c = "COMO DEVO USAR ESTE MEDICAMENTO?"
            
            if query1c in result_q1[1].page_content:
                result_q1 = result_q1[1].page_content
                segundo += 1
                count += 1
            else:
                result_q1 = result_q1[0].page_content
                primeiro += 1
                if query1c in result_q1:
                    count += 1
                else:
                    false += 1

            #print(f"{result_q1}\n\n\n")
            
            df.loc[len(df)] = [file_path[25:-4], result_q1, result_q2[0].page_content]
            print(df.loc[0]['item 6'])
            print()
            print(df.loc[0]['item7'])
            print(f"{count} / {total}\n")
    print(f"\n\n========  {primeiro} / {segundo}  ==========")
        #df.to_csv("teste_erro.csv", index=False)
    df.to_csv(f"RAG_teste.csv", index=False)

main()
