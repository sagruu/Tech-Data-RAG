import logging
import os
import pathlib
import time

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from dotenv import load_dotenv
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")


class RAGAS_Evaluator:
    def __init__(self, embedding_deployment=str, llm_deployment="gpt-4o-mini"):
        self.api_key = AZURE_OPENAI_API_KEY
        self.endpoint = AZURE_OPENAI_ENDPOINT
        self.embedding_deployment = embedding_deployment
        self.llm_deployment = llm_deployment
        self.data_path = "../data_mc1/data_processed/cleantech_rag_evaluation_data.parquet"

    def load_test_data(self):
        df = pd.read_parquet(self.data_path)
        required_columns = ["question", "relevant_text", "answer"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Fehlende Spalten: {missing_columns}")
        # Entferne Zeilen mit leeren oder NaN-Werten in den erforderlichen Spalten
        df = df.dropna(subset=required_columns)
        df = df[df[required_columns].ne("").all(axis=1)]
        return df

    def load_vectorstore(self, collection_dir_name):
        embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment=self.embedding_deployment,
            chunk_size=1
        )

        persist_directory = pathlib.Path.cwd().parent / "src" / collection_dir_name
        collection_name = collection_dir_name.replace("chroma_", "")

        client = chromadb.PersistentClient(path=str(persist_directory))
        existing_collections = client.list_collections()  # Gibt jetzt eine Liste von Namen zurück

        if collection_name not in existing_collections:
            raise ValueError(f"Collection '{collection_name}' nicht gefunden in {persist_directory}.")

        logging.info(f"Collection '{collection_name}' gefunden in {persist_directory}.")
        return Chroma(
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=str(persist_directory)
        )

    def get_retrievers(self, vector_store):
        return {
            "similarity_k3": vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            "similarity_k5": vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            "mmr_k5": vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
            "mmr_k7": vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 7, "fetch_k": 20}),
        }

    def get_retriever(self, collection_name, retriever_name):
        vs = self.load_vectorstore(collection_name)
        return self.get_retrievers(vs)[retriever_name]

    def get_prompts(self):
        return {
            "simple": ChatPromptTemplate.from_template(
                """Answer the following question based on the context provided.
                Your task is to answer concisely and accurately.
                – For yes/no questions: reply only with "yes" or "no".
                – For questions targeting a specific named entity: reply only with the named entity (e.g., a date, person, place, or organization) and nothing else.
                Do not add any justification or background information in either case.
                Context:
                {context}
                Question:
                {question}
                If you cannot find an answer, please state that you do not know."""
            ),
            "chain_of_thought": ChatPromptTemplate.from_template(
                """Given the context below, think carefully and step-by-step before answering the question.
                Context:
                {context}
                Question:
                {question}
                Please explain your reasoning before giving a final answer.
                If the answer cannot be found explicitly, clearly state that."""
            ),
        }

    def get_prompt(self, prompt_name):
        return self.get_prompts()[prompt_name]

    def get_chain(self, prompt, retriever, temperature=0.0):
        return self.build_chain(retriever, prompt, temperature)

    def build_chain(self, retriever, prompt_template, temperature):
        llm = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment_name=self.llm_deployment,
            temperature=temperature
        )

        return (
            RunnableMap({
                "retrieved_documents": lambda x: retriever.invoke(x["question"]) or [],
                "question": lambda x: x["question"]
            }) | RunnableMap({
                "retrieved_texts": lambda x: [doc.page_content for doc in x["retrieved_documents"]],
                "question": lambda x: x["question"]
            }) | RunnableMap({
                "context": lambda x: "\n\n".join(x["retrieved_texts"]),
                "retrieved_texts": lambda x: x["retrieved_texts"],
                "question": lambda x: x["question"]
            }) | RunnableMap({
                "llm_response": lambda x: llm.invoke(
                    prompt_template.format_messages(context=x['context'], question=x['question'])
                ),
                "retrieved_texts": lambda x: x["retrieved_texts"],
                "context": lambda x: x["context"],
                "question": lambda x: x["question"]
            })
        )

    def run_batch(self, df, chain):
        results = []
        for idx, row in df.iterrows():
            try:
                output = chain.invoke({"question": row["question"]})
            except Exception as e:
                logging.error(f"Fehler bei Frage '{row['question']}': {e}")
                output = {"retrieved_texts": [], "llm_response": "Fehler", "context": "", "question": row["question"]}
            
            time.sleep(1)  # Rate Limit

            results.append({
                "question": row["question"],
                "relevant_text": row["relevant_text"],
                "relevant_text_llm": output["retrieved_texts"],  # Speichere als Liste
                "retrieved_context": output["context"],
                "answer": row.get("answer", ""),
                "answer_llm": output["llm_response"].content if hasattr(output["llm_response"], "content") else output["llm_response"]
            })

            if idx % 5 == 0:
                logging.info(f"{idx}/{len(df)} Fragen evaluiert...")

        return pd.DataFrame(results)

    def run_ragas(self, df_result):
        df_ragas_input = df_result.rename(columns={
            "relevant_text_llm": "retrieved_contexts",
            "relevant_text": "reference"
        })[["question", "retrieved_contexts", "answer_llm", "answer", "reference"]]

        # Stelle sicher, dass retrieved_contexts eine Liste ist
        df_ragas_input["retrieved_contexts"] = df_ragas_input["retrieved_contexts"].apply(
            lambda x: x if isinstance(x, list) else [x]
        )

        for col in ["question", "answer_llm", "answer", "reference"]:
            df_ragas_input[col] = df_ragas_input[col].astype(str)

        ragas_dataset = Dataset.from_pandas(df_ragas_input)

        embedding = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment=self.embedding_deployment,
            chunk_size=1
        )
        llm = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment_name=self.llm_deployment,
            temperature=0.0
        )

        return evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
            embeddings=embedding,
            llm=llm
        )

    def is_similar(self, texts_a, texts_b, threshold=0.5):
        embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment=self.embedding_deployment,
            chunk_size=1
        )

        # Konvertiere Eingaben in Listen, falls sie es nicht sind
        texts_a = texts_a if isinstance(texts_a, list) else [texts_a]
        texts_b = texts_b if isinstance(texts_b, list) else [texts_b]

        try:
            # Batch Embedding-Berechnung
            emb_a = embedding_model.embed_documents(texts_a)
            emb_b = embedding_model.embed_documents(texts_b)

            # Berechne Cosinus-Ähnlichkeit für alle Paare
            similarities = []
            for a, b in zip(emb_a, emb_b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a * norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(a, b) / (norm_a * norm_b)
                similarities.append(similarity)

            # Gib True zurück, wenn irgendeine Ähnlichkeit den Schwellenwert überschreitet
            return any(sim > threshold for sim in similarities)

        except Exception as e:
            print(f"Fehler bei der Ähnlichkeitsberechnung: {e}")
            return False

    def evaluate_ir_metrics(self, df_result, k=5, similarity_threshold=0.5):
        precision_scores, recall_scores, reciprocal_ranks = [], [], []

        for _, row in df_result.iterrows():
            # Abgerufene Kontexte vorbereiten
            retrieved = row["relevant_text_llm"]
            retrieved = retrieved if isinstance(retrieved, list) else retrieved.split("\n\n")
            # Entferne Duplikate und begrenze auf k
            retrieved = list(dict.fromkeys(retrieved))[:k]

            # Gold-Kontexte vorbereiten (als Liste, auch wenn es nur ein Element ist)
            gold = row["relevant_text"]
            gold = gold if isinstance(gold, list) else [gold]

            # Prüfe, welche abgerufenen Kontexte relevant sind
            hits = []
            for chunk in retrieved:
                is_relevant = self.is_similar(gold, [chunk], threshold=similarity_threshold)
                hits.append(1 if is_relevant else 0)

            # Precision: Anteil der Top-k Kontexte, die relevant sind
            precision = sum(hits) / k if retrieved else 0.0

            # Recall: Anteil der Gold-Kontexte, die in den Top-k gefunden wurden
            matched_gold = set()
            for i, chunk in enumerate(retrieved):
                if hits[i]:
                    for j, g in enumerate(gold):
                        if self.is_similar([g], [chunk], threshold=similarity_threshold):
                            matched_gold.add(j)
            recall = len(matched_gold) / len(gold) if gold else 0.0

            # MRR: Reziproker Rang des ersten Treffers
            rr = next((1.0 / (i + 1) for i, chunk in enumerate(retrieved)
                       if self.is_similar(gold, [chunk], threshold=similarity_threshold)), 0.0)

            precision_scores.append(precision)
            recall_scores.append(recall)
            reciprocal_ranks.append(rr)

        return {
            f"Precision@{k}": round(np.mean(precision_scores), 4),
            f"Recall@{k}": round(np.mean(recall_scores), 4),
            "MRR": round(np.mean(reciprocal_ranks), 4)
        }

    def compute_similarity_to_gold(self, df_result):
        embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2023-05-15",
            deployment=self.embedding_deployment,
            chunk_size=1
        )

        llm_answers = df_result["answer_llm"].astype(str).tolist()
        gold_answers = df_result["answer"].astype(str).tolist()

        try:
            llm_embs = embedding_model.embed_documents(llm_answers)
            gold_embs = embedding_model.embed_documents(gold_answers)
            similarities = [
                np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                if np.linalg.norm(emb_a) * np.linalg.norm(emb_b) != 0 else 0.0
                for emb_a, emb_b in zip(llm_embs, gold_embs)
            ]
        except Exception as e:
            print(f"Fehler bei der Ähnlichkeitsberechnung: {e}")
            similarities = [0.0] * len(df_result)

        df_result["similarity_to_gold"] = similarities
        return df_result

    def plot_eval_result(self, df: pd.DataFrame):
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        columns = [col for col in [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness", "similarity_to_gold"
        ] if col in df.columns]

        if not columns:
            raise ValueError("Keine der erwarteten RAGAS-Metriken im DataFrame gefunden.")

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[columns], palette="Set1", width=0.6, linewidth=1.5)
        plt.title(f"{getattr(self, 'name', 'Evaluation')}: RAGAS Metrics Boxplot", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

    def plot_eval_result_bar(self, df: pd.DataFrame):
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        all_metrics = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness", "similarity_to_gold",
            "MRR", "Precision@5", "Recall@5"
        ]
        available = [m for m in all_metrics if m in df.columns]
        if not available:
            raise ValueError("Keine passenden Metriken für Barplot gefunden.")

        metric_means = df[available].mean()

        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=metric_means.index,
            y=metric_means.values,
            palette="Set1"
        )
        plt.title(f"{getattr(self, 'name', 'Evaluation')}: RAGAS + Non-LLM Metrics (Mean)", fontsize=16)
        plt.ylabel("Mean Score", fontsize=14)
        plt.xlabel("Metrics", fontsize=14)
        plt.xticks(fontsize=12, rotation=20)
        plt.tight_layout()
        plt.show()

    def plot_results_all(self, df: pd.DataFrame):
        self.plot_eval_result(df)
        self.plot_eval_result_bar(df)

    def run(self, collection_name: str, retriever_name: str, prompt_name: str, temperature: float = 0.0, save_path: str = None, plot: bool = True):
        df = self.load_test_data()
        retriever = self.get_retriever(collection_name, retriever_name)
        prompt = self.get_prompt(prompt_name)
        chain = self.get_chain(prompt, retriever, temperature=temperature)

        results_df = self.run_batch(df, chain)
        ragas_metrics = self.run_ragas(results_df)
        ragas_df = ragas_metrics.to_pandas()

        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']:
            if metric in ragas_df.columns:
                results_df[metric] = ragas_df[metric].values

        numeric_df = ragas_df.select_dtypes(include="number")
        ragas_dict = numeric_df.mean().to_dict()

        ir_metrics = self.evaluate_ir_metrics(results_df)
        for key, value in ir_metrics.items():
            results_df[key] = value

        if plot:
            self.plot_results_all(results_df)

        if save_path:
            results_df.to_csv(save_path, index=False)

        return results_df, ragas_dict, ir_metrics