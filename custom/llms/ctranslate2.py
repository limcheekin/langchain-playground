# %% [markdown]
# # How to write a custom LLM wrapper
#
# This notebook goes over how to create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper than one that is supported in LangChain.
#
# There is only one required thing that a custom LLM needs to implement:
#
# 1. A `_call` method that takes in a string, some optional stop words, and returns a string
#
# There is a second optional thing it can implement:
#
# 1. An `_identifying_params` property that is used to help with printing of this class. Should return a dictionary.
#
# Let's implement a very simple custom LLM that just returns the first N characters of the input.

# %%
from typing import Any, Union, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import ctranslate2
import transformers
# %%


class Ct2Translator(LLM):
    """Wrapper around ctranslate2.Translator class.

    Example:
        .. code-block:: python

            from ctranslate2.llms import Ct2Translator
            translator = Ct2Translator(model_path="./ct2fast-flan-alpaca-xl")
    """
    model_path: str = None
    n_threads: int = 1          # inter_threads
    beam_size: int = 2
    top_k: int = 1              # sampling_topk
    temperature: float = 1      # sampling_temperature
    repeat_penalty: float = 1   # repetition_penalty
    no_repeat_ngram_size: int = 0
    min_length: int = 1         # min_decoding_length
    max_length: int = 256       # max_decoding_length

    translator: ctranslate2.Translator = None
    tokenizer: transformers.AutoTokenizer = None

    def init(self):
        self.translator = ctranslate2.Translator(
            model_path=self.model_path,
            inter_threads=self.n_threads,
            compute_type="int8"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path)

    @property
    def _llm_type(self) -> str:
        return "Ct2Translator"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt))
        results = self.translator.translate_batch(
            [input_tokens],
            beam_size=self.beam_size,
            sampling_topk=self.top_k,
            sampling_temperature=self.temperature,
            repetition_penalty=self.repeat_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            min_decoding_length=self.min_length,
            max_decoding_length=self.max_length,
        )
        output_tokens = results[0].hypotheses[0]
        output_text = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens))
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self._default_params()

    def _default_params(self) -> Mapping[str, Any]:
        return {
            "n_threads": self.n_threads,
            "beam_size": self.beam_size,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "temperature": self.temperature,
        }

# %% [markdown]
# We can now use this as an any other LLM.
