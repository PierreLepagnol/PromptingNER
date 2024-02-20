import os
import random
from typing import Tuple

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_tokenizer(model_name: str, trust_remote_code: bool = False) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, model_type: str, trust_remote_code: bool = False) -> PreTrainedModel:
    print(f"Loading model {model_name}")

    if model_type == "causal":
        return AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif model_type == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        raise NotImplementedError("Invalid model_type. Choose from 'causal' or 'seq2seq'")


def load_model_and_tokenizer(
    model_name: str,
    model_type: str = "causal",
    trust_remote_code: bool = False,
    debug: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and its tokenizer from Hugging Face's model hub.

    This function supports both causal language models and sequence-to-sequence models.
    The specific type of model that is loaded depends on the `model_name` parameter.

    If `model_name` is "bigscience/bloomz-560m", a causal language model is loaded.
    Otherwise, a sequence-to-sequence model is loaded.

    After the model is loaded, it is set up using the provided `fabric` object.

    Args:
        fabric (Fabric): The fabric object to use for setting up the model.
        model_name (str, optional): The name of the model to load.
            Defaults to "bigscience/bloomz-560m".

    Returns:
        tuple: A tuple containing the loaded model and its tokenizer.
    """

    if debug:
        with init_empty_weights():
            model = load_model(model_name, model_type, trust_remote_code=trust_remote_code)
    else:
        model = load_model(model_name, model_type, trust_remote_code=trust_remote_code)

    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def load_template_creator(model_name: str):
    if model_name == "stabilityai/stablelm-2-zephyr-1_6b":
        from utils.datasets import StableLMZephyrTemplate

        return StableLMZephyrTemplate

    elif model_name == "stabilityai/stablelm-2-1_6b":
        from utils.datasets import StableLMTemplate

        return StableLMTemplate

    elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        from utils.datasets import MistralInstructTemplate

        return MistralInstructTemplate
    else:
        raise NotImplementedError("Template creator not implemented for this model")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
