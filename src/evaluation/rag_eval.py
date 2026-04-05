import asyncio
from utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy


def evaluate_context_precision(query: str, response: str, retrieved_context: list) -> float:
    """
    Evaluate the context precision metric for a given query-response pair.

    This function measures the precision of the retrieved context using the
    LLMContextPrecisionWithoutReference metric from RAGAS.

    Args:
        query (str): The user query.
        response (str): The generated LLM response.
        retrieved_context (list): List of retrieved contexts used for RAG.

    Returns:
        float: The context precision score (0 to 1).
    
    Raises:
        Exception: In case of evaluation failure.
    """
    try:
        # Prepare the sample for evaluation
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        async def main():
            # Load a language model and wrap it
            llm = ModelLoader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)

            # Initialize the Context Precision evaluator
            context_precision_metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

            # Compute precision score
            score = await context_precision_metric.single_turn_ascore(sample)
            return score

        # Run the async evaluation
        return asyncio.run(main())

    except Exception as e:
        return e


def evaluate_response_relevancy(query: str, response: str, retrieved_context: list) -> float:
    """
    Evaluate the response relevancy of a generated answer to a user query.

    This function measures how relevant the answer is to the query in the context
    of the retrieved documents using the ResponseRelevancy metric from RAGAS.

    Args:
        query (str): The original user query.
        response (str): The generated LLM response.
        retrieved_context (list): List of retrieved contexts used for RAG.

    Returns:
        float: The response relevancy score (0 to 1).
    
    Raises:
        Exception: In case of evaluation failure.
    """
    try:
        # Prepare the sample for evaluation
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )

        async def main():
            # Load LLM for scoring
            llm = ModelLoader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)

            # Load embedding model for semantic comparison
            embedding_model = ModelLoader.load_embeddings()
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

            # Initialize relevancy scorer
            relevancy_scorer = ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )

            # Compute relevancy score
            score = await relevancy_scorer.single_turn_ascore(sample)
            return score

        # Execute the async evaluator
        return asyncio.run(main())

    except Exception as e:
        return e
