import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.query_pipeline import FnComponent, QueryPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from tqdm import tqdm


class TextToSQLPipeline:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        embedding_model: str,
        base_directory: str,
        similarity_top_k: int = 3,
        max_workers: int = 4,
    ):
        """
        Initializes the TextToSQLPipeline with LLM, embedding model, and schema-specific indices.

        :param model_name: Name of the LLM model to use.
        :param base_url: Base URL for the LLM service.
        :param embedding_model: HuggingFace embedding model name.
        :param base_directory: Base directory containing schema directories.
        :param similarity_top_k: Number of top similar documents to retrieve.
        :param max_workers: Maximum number of threads for building schema indices.
        """
        self.llm = Ollama(model=model_name, base_url=base_url)
        self.similarity_top_k = similarity_top_k
        self.max_workers = max_workers
        self.schema_indices = self._build_schema_indices(base_directory, embedding_model)
        self.pipeline = self._build_pipeline()

    def _build_schema_indices(
        self, base_directory: str, embedding_model: str
    ) -> Dict[str, VectorStoreIndex]:
        """
        Builds schema indices with a progress bar.

        :param base_directory: Directory containing schema folders.
        :param embedding_model: HuggingFace embedding model name.
        :return: Dictionary mapping schema names to their indices.
        """
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        schema_indices = {}
        schema_dirs = [
            schema_name
            for schema_name in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, schema_name))
        ]

        def process_schema(schema_name: str) -> tuple[str, VectorStoreIndex | None]:
            """
            Processes a single schema directory to build its index.

            :param schema_name: Name of the schema directory.
            :return: Tuple of schema name and its corresponding VectorStoreIndex.
            """
            schema_path = os.path.join(base_directory, schema_name)
            documents = SimpleDirectoryReader(schema_path).load_data()
            index = VectorStoreIndex.from_documents(documents)
            return schema_name, index

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_schema, schema_name): schema_name
                for schema_name in schema_dirs
            }

            # Use tqdm to track progress
            for future in tqdm(
                as_completed(futures), total=len(schema_dirs), desc="Indexing schemas"
            ):
                schema_name = futures[future]
                try:
                    result = future.result()
                    if result[1] is not None:  # Only add if the index was built successfully
                        schema_indices[result[0]] = result[1]
                except Exception as e:
                    print(f"Error processing schema '{schema_name}': {e}")

        return schema_indices

    def _build_pipeline(self) -> QueryPipeline:
        """Builds the query pipeline with retrieval and generation components."""
        retrieve_component = FnComponent(
            fn=self._retrieve_context,
            input_keys=["query_str"],
            output_keys=["context", "query_str"]
        )
        generate_component = FnComponent(
            fn=self._generate_sql,
            input_keys=["context", "query_str"],
            output_keys=["sql_query"]
        )
        return QueryPipeline(chain=[retrieve_component, generate_component])

    def _retrieve_contexts(self, query_str: str, schema_names: list) -> str:
        """
        Retrieves combined context from multiple schemas.

        :param query_str: The query string.
        :param schema_names: List of schema names to query.
        :return: Combined context string.
        """
        contexts = []
        for schema_name in schema_names:
            if schema_name not in self.schema_indices:
                print(f"Warning: Schema '{schema_name}' not found.")
                continue
            retriever = self.schema_indices[schema_name].as_retriever(
                similarity_top_k=self.similarity_top_k
            )
            nodes = retriever.retrieve(query_str)
            schema_context = " ".join([node.get_content() for node in nodes])
            contexts.append(f"Schema: {schema_name}\n{schema_context}")
        return "\n\n".join(contexts)

    def _generate_sql(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Generates an SQL query based on combined context and query.

        :param input_dict: Dictionary containing the combined context and query string.
        :return: Generated SQL query.
        """
        context = input_dict["context"]
        query_str = input_dict["query_str"]
        prompt_template = PromptTemplate(
            "Context information from multiple schemas is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and no other information, generate a SQL query to answer the question: {query_str}\n"
            "SQL Query:"
        )
        prompt = prompt_template.format(context_str=context, query_str=query_str)
        response = self.llm.complete(prompt)
        return {"sql_query": response.text}

    def run_multi_schema_query(self, query_str: str, schema_names: list) -> Dict[str, Any]:
        """
        Runs a multi-schema query by combining contexts from specified schemas.

        :param query_str: The query string.
        :param schema_names: List of schema names to query.
        :return: The resulting SQL query.
        """
        try:
            combined_context = self._retrieve_contexts(query_str, schema_names)
            result = self.pipeline.run({"context": combined_context, "query_str": query_str})
            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred during query processing: {e}") from e


# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "duckdb-nsql"
    BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "s2593817/sft-sql-embedding"
    BASE_DIRECTORY = "yaml_docs"

    # Initialize the pipeline
    pipeline = TextToSQLPipeline(
        model_name=MODEL_NAME,
        base_url=BASE_URL,
        embedding_model=EMBEDDING_MODEL,
        base_directory=BASE_DIRECTORY,
    )

    query = "Get email templates and their associated transactions created in the last 180 days."
    schema_names = ["crm_etl", "marketing_etl"]  # Specify schemas to query

    try:
        result = pipeline.run_multi_schema_query(query_str=query, schema_names=schema_names)
        print(result["sql_query"])
    except Exception as e:
        print(f"Error: {e}")
