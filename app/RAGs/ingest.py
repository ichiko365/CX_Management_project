import json
import os
from dotenv import load_dotenv
from pathlib import Path
try:
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm  # Import tqdm for progress bars

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

# 1. Create a path to a new cache directory inside your project
CACHE_DIR = BASE_DIR / ".huggingface_cache"

# 2. Make sure the directory actually exists
CACHE_DIR.mkdir(exist_ok=True)

# 3. Set HF_HOME to this new, portable path
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# make the current folder the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = "../data/product/description.json"
QA_DB_PATH = ".chroma_qa"
CATALOG_DB_PATH = ".chroma_catalog"
FAQ_DB_PATH = ".chroma_faqs"
FAQS_JSON_PATH = "../data/FAQs.json"


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
    # Use a lightweight CPU embedding model to avoid MPS OOM
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "mps"},
        encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
    )
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

    # FAQs Index (optional)
    try:
        if os.path.exists(FAQS_JSON_PATH):
            with open(FAQS_JSON_PATH, "r", encoding="utf-8") as f:
                faqs_data = json.load(f)
            texts = []
            metas = []
            if isinstance(faqs_data, dict):
                for q, a in faqs_data.items():
                    if isinstance(a, str):
                        texts.append(f"Q: {q}\nA: {a}")
                        metas.append({"type": "faq"})
            elif isinstance(faqs_data, list):
                for it in faqs_data:
                    if isinstance(it, dict):
                        q = it.get("question") or it.get("q") or it.get("Question")
                        a = it.get("answer") or it.get("a") or it.get("Answer")
                        if q and a:
                            texts.append(f"Q: {q}\nA: {a}")
                            metas.append({"type": "faq"})
            if texts:
                build_index(texts, metas, FAQ_DB_PATH, "FAQs")
    except Exception as e:
        print(f"Skipping FAQs index build due to error: {e}")
    
    print("\nAll indexes have been built successfully!")


if __name__ == "__main__":
    # Ensure you have tqdm installed: pip install tqdm
    build_indexes()