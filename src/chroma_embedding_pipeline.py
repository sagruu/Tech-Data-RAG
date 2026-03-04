
import os
import time
from typing import Optional, Dict

import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings


class ChromaEmbeddingPipeline:
    """
    Pipeline zur Erstellung und Speicherung von Embeddings in einer Chroma-Datenbank.

    Parameter:
        df (pd.DataFrame): Daten
        text_column (str): Spaltenname, aus welcher Embeddings erzeugt werden sollen
        deployment_name (str): Azure-Deployment-Name des Embedding-Modells
        splitter (TextSplitter): Beliebiger LangChain TextSplitter
        collection_name (str): Name der Chroma Collection (wird automatisch generiert, wenn "auto" übergeben)
        persist_directory (str): Speicherort für die DB (wird automatisch generiert, wenn "auto" übergeben)
        batch_size (int): Größe der Embedding-Batches
        sleep_time (float): Wartezeit in Sekunden zwischen Batches
        metadata (dict): Optionales Metadaten-Dict für die Chroma Collection
        chroma_settings (ChromaSettings): Chroma-Client-Einstellungen
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str,
        deployment_name: str,
        splitter: Optional[TextSplitter] = None,
        collection_name: str = "auto",
        persist_directory: str = "auto",
        batch_size: int = 10,
        sleep_time: float = 1.0,
        metadata: Optional[Dict] = None,
        chroma_settings: Optional[ChromaSettings] = None
    ):
        if text_column not in df.columns:
            raise ValueError(f"Spalte '{text_column}' nicht im DataFrame gefunden.")

        self.df = df
        self.text_column = text_column
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        
        # Standard-Splitter, falls keiner übergeben wird
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Falls Collection-Name nicht angegeben, wird er automatisch generiert
        if collection_name == "auto":
            self.collection_name = self._generate_collection_name(deployment_name)
        else:
            self.collection_name = collection_name

        # Falls persist_directory nicht angegeben, wird automatisch generiert
        if persist_directory == "auto":
            self.persist_directory = f"./chroma_{self.collection_name}"
        else:
            self.persist_directory = persist_directory

        # Standard-Metadaten, falls keine übergeben werden
        self.metadata = metadata or {"hnsw:space": "cosine"}

        # Initialisiere das Azure-OpenAI-Embedding-Modell
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # in ENV setzen
            api_key=os.environ["AZURE_OPENAI_API_KEY"],          # in ENV setzen
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # in ENV setzen
            deployment=deployment_name
        )

        # Chroma-Einstellungen (persist_directory etc.)
        self.chroma_settings = chroma_settings or ChromaSettings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )

        # Chroma-Client erstellen
        self.client = chromadb.Client(settings=self.chroma_settings)

        # Falls Collection noch nicht vorhanden, wird sie neu angelegt
        existing_collections = self.client.list_collections()
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                name=self.collection_name,
                metadata=self.metadata
            )

    def _generate_collection_name(self, deployment_name: str) -> str:
        """
        Generiert einen Collection-Namen basierend auf dem TextSplitter und dem Deployment-Namen.

        Achtung: Wir greifen hier auf die internen Attribute _chunk_size und _chunk_overlap zu,
        da neuere LangChain-Versionen chunk_size/chunk_overlap nicht mehr öffentlich bereitstellen.
        """
        name = type(self.splitter).__name__
        name = name.lower().replace("textsplitter", "") \
            .replace("character", "char") \
            .replace("recursive", "rec") \
            .replace(" ", "")
        
        # Beachte: _chunk_size und _chunk_overlap sind "private"
        # und könnten sich in zukünftigen LangChain-Versionen ändern.
        cs = getattr(self.splitter, "_chunk_size", "NA")
        co = getattr(self.splitter, "_chunk_overlap", "NA")

        # "text-embedding-" entfernen, um den Deployment-Namen etwas abzukürzen
        model_short = deployment_name.replace("text-embedding-", "")

        return f"{name}_cs{cs}_co{co}_{model_short}"

    def run(self) -> Chroma:
        """
        Führt den vollständigen Embedding-Prozess aus und speichert alles lokal
        in der Chroma-Datenbank ab. Gibt anschließend das Chroma-Objekt zurück.
        """
        # 1) Textdaten holen & Duplikate entfernen
        texts = (
            self.df[self.text_column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        # 2) Chunks erzeugen
        chunks = []
        for text in texts:
            chunks.extend(self.splitter.split_text(text))
        print(f"{len(chunks)} Chunks erzeugt.")

        # 3) Chroma-Objekt erstellen
        vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        # 4) Chunks in Batches verarbeiten
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                docs = [Document(page_content=chunk) for chunk in batch]
                vector_store.add_documents(docs)
            except Exception as e:
                print(f"Fehler bei Batch {i // self.batch_size + 1}: {e}")
                continue

            print(f"Batch {i // self.batch_size + 1}: {len(batch)} Chunks hinzugefügt")
            time.sleep(self.sleep_time)

        # 5) Datenbank persistieren
        vector_store.persist()
        print(f"Gespeichert unter: {self.persist_directory}")
        return vector_store
