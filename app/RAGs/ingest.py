import json
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm  # Import tqdm for progress bars

load_dotenv()

# make the current folder the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = "../data/product/description.json"
QA_DB_PATH = ".chroma_qa"
CATALOG_DB_PATH = ".chroma_catalog"


def load_data(file_path: str):
    """Load product data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)
    print("Data loaded successfully.")
    return data


def prepare_qa_data(data):
    """Prepare long text entries for QA index with a progress bar."""
    texts, metadatas = [], []
    # Wrap the data iterator with tqdm to show progress
    for item in tqdm(data, desc="Preparing QA data"):
        asin = item.get("ASIN", "")
        title = item.get("Title", "")
        desc = item.get("Description", "")

        qa_text = f"Title: {title}\nDescription: {desc}"
        texts.append(qa_text)
        metadatas.append({"ASIN": asin, "Title": title})

    return texts, metadatas


def prepare_catalog_data(data):
    """Prepare short text entries for Catalog index with a progress bar."""
    texts, metadatas = [], []
    # Wrap the data iterator with tqdm to show progress
    for item in tqdm(data, desc="Preparing Catalog data"):
        asin = item.get("ASIN", "")
        title = item.get("Title", "")
        desc = item.get("Description", "")

        short_desc = " ".join(desc.split()[:20])
        catalog_text = f"Title: {title}\nShortDesc: {short_desc}"
        texts.append(catalog_text)
        metadatas.append({"ASIN": asin, "Title": title})

    return texts, metadatas


def build_index(texts, metadatas, persist_path: str, index_name: str):
    """Build and persist a Chroma index."""
    print(f"\nBuilding {index_name} index...")
    # Note: Chroma.from_texts is a bulk operation. 
    # A progress bar is not easily applicable here without batching.
    # The console messages will indicate the start and end of this process.
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    db = Chroma.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory=persist_path
    )
    print(f"{index_name} index built and persisted to {persist_path}.")


def build_indexes():
    """Main pipeline to build both QA and Catalog indexes."""
    data = load_data(DATA_PATH)

    # QA Index
    qa_texts, qa_metadatas = prepare_qa_data(data)
    build_index(qa_texts, qa_metadatas, QA_DB_PATH, "QA")

    # Catalog Index
    catalog_texts, catalog_metadatas = prepare_catalog_data(data)
    build_index(catalog_texts, catalog_metadatas, CATALOG_DB_PATH, "Catalog")
    
    print("\nAll indexes have been built successfully!")


if __name__ == "__main__":
    # Ensure you have tqdm installed: pip install tqdm
    build_indexes()