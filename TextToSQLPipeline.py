import os
from typing import Dict, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_pipeline import FnComponent, QueryPipeline
from llama_index.llms.ollama import Ollama


class TextToSQLPipeline:
    def __init__(self, 
                 model_name: str, 
                 base_url: str, 
                 embedding_model: str, 
                 directory: str, 
                 similarity_top_k: int = 3):
        """
        Initializes the TextToSQLPipeline with LLM, embedding model, and index configuration.

        :param model_name: Name of the LLM model to use.
        :param base_url: Base URL for the LLM service.
        :param embedding_model: HuggingFace embedding model name.
        :param directory: Directory containing the documents to index.
        :param similarity_top_k: Number of top similar documents to retrieve.
        """
        self.llm = Ollama(model=model_name, base_url=base_url)
        self.similarity_top_k = similarity_top_k
        self.index = self._build_index(directory, embedding_model)
        self.pipeline = self._build_pipeline()

    @staticmethod
    def _build_index(directory: str, embedding_model: str) -> VectorStoreIndex:
        """Builds the document index using the specified embedding model and directory."""
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        documents = SimpleDirectoryReader(directory).load_data()
        return VectorStoreIndex.from_documents(documents)

    def _retrieve_context(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """Retrieves context from the document index based on the input query."""
        query_str = input_dict["query_str"]
        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(query_str)
        context = " ".join([node.get_content() for node in nodes])
        return {"context": context, "query_str": query_str}

    def _generate_sql(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """Generates an SQL query based on the provided context and query."""
        context = input_dict["context"]
        query_str = input_dict["query_str"]
        prompt_template = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and no other information, generate a SQL query to answer the question: {query_str}\n"
            "SQL Query:"
        )
        prompt = prompt_template.format(context_str=context, query_str=query_str)
        response = self.llm.complete(prompt)
        return {"sql_query": response.text}

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

    def run_query(self, query_str: str) -> Dict[str, Any]:
        """Runs the query pipeline and returns the SQL query."""
        try:
            result = self.pipeline.run({"query_str": query_str})
            return result
        except KeyError as e:
            raise ValueError(f"Missing key in pipeline result: {e}") from e
        except Exception as e:
            raise RuntimeError(f"An error occurred during query processing: {e}") from e


# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "duckdb-nsql"
    BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "s2593817/sft-sql-embedding"
    DOCUMENTS_DIR = "yaml_docs"

    # Initialize and run the pipeline
    pipeline = TextToSQLPipeline(
        model_name=MODEL_NAME,
        base_url=BASE_URL,
        embedding_model=EMBEDDING_MODEL,
        directory=DOCUMENTS_DIR
    )

    query = "Published 'TRANSACTIONAL' pigeon email templates dispatched (created_date) in last 180 days with more than 3 substitution variables"
    try:
        result = pipeline.run_query(query)
        print(result["sql_query"])
    except Exception as e:
        print(f"Error: {e}")
