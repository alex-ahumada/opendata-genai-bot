MODEL_COST_PER_1K_TOKENS = {
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0613": 0.06,
    "gpt-4-1106-preview": 0.01,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    "gpt-4-1106-preview-completion": 0.03,
    # GPT-3.5 input
    "gpt-3.5-turbo": 0.001,
    "gpt-3.5-turbo-0613": 0.001,
    "gpt-3.5-turbo-1106": 0.001,
    "gpt-3.5-turbo-instruct": 0.0015,
    "gpt-3.5-turbo-16k": 0.001,
    "gpt-3.5-turbo-16k-0613": 0.001,
    # GPT-3.5 output
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-1106-completion": 0.002,
    "gpt-3.5-turbo-instruct-completion": 0.003,
    "gpt-3.5-turbo-16k-completion": 0.002,
    "gpt-3.5-turbo-16k-0613-completion": 0.002,
    # Azure GPT-35 input
    "gpt-35-turbo": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613": 0.0015,
    "gpt-35-turbo-instruct": 0.0015,
    "gpt-35-turbo-16k": 0.003,
    "gpt-35-turbo-16k-0613": 0.003,
    # Azure GPT-35 output
    "gpt-35-turbo-completion": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613-completion": 0.002,
    "gpt-35-turbo-instruct-completion": 0.002,
    "gpt-35-turbo-16k-completion": 0.004,
    "gpt-35-turbo-16k-0613-completion": 0.004,
    # Fine-tuned input
    "gpt-3.5-turbo-0613-finetuned": 0.012,
    # Fine-tuned output
    "gpt-3.5-turbo-0613-finetuned-completion": 0.016,
    # Azure Fine-tuned output
    "gpt-35-turbo-0613-azure-finetuned": 0.0015,
    # Azure Fine-tuned output
    "gpt-35-turbo-0613-azure-finetuned-completion": 0.002,
    # Others
    "text-davinci-003": 0.02,
}


def get_openai_token_cost_for_model(
    model_name: str,
    num_tokens: int,
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name (str): Name of the model
        num_tokens (int): Number of tokens.

    Returns:
        float: Cost in USD.
    """
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(f"Unknown model: {model_name}.")
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)
