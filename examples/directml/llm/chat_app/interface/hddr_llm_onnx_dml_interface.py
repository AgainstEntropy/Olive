import gc
import os

import numpy as np
import onnxruntime
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state
from interface.base_interface import BaseLLMInterface
from transformers import AutoTokenizer


class LLMOnnxDmlInterface(BaseLLMInterface):
    def __init__(self, model_dir=""):
        super().__init__()

        self.model_dir = model_dir
        self.total_count = 0

    def initialize(self):
        # Create the ONNX sessions

        providers = [
            (
                "DmlExecutionProvider",
                {
                    "disable_metacommands": True,
                    "enable_dynamic_graph_fusion": True,
                },
            )
        ]

        llm_session_options = onnxruntime.SessionOptions()
        llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
        llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)

        self.llm_session = onnxruntime.InferenceSession(
            os.path.join(self.model_dir, "decoder_model_merged.onnx"),
            sess_options=llm_session_options,
            providers=providers,
        )

        self.data_type = np.float16
        max_seq_len = 2048
        self.num_layers = 0
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name.startswith("cache.") and inputs_meta.name.endswith(".key"):
                self.num_layers += 1
                num_key_value_heads = inputs_meta.shape[1]
                head_dim = inputs_meta.shape[3]

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self.binding_device = "dml"

        # Create the I/O bindings
        self.llm_io_binding = self.llm_session.io_binding()

        # Initialize the buffers
        self.tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, self.binding_device)

        # Create the K and V caches.
        cache_shape = (1, num_key_value_heads, max_seq_len, head_dim)
        initial_cache = np.zeros(cache_shape, dtype=self.data_type)
        self.k_caches = []
        self.v_caches = []

        for _ in range(self.num_layers):
            self.k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.binding_device))
            self.v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.binding_device))

        self.initial_prompt = [
            {"role": "user", "content": "Hey there I am a human that would like to have a conversation with you."},
            {"role": "assistant", "content": "Sure, I am happy to answer most questions."},
            {"role": "user", "content": "Great, I insist that we take turns."},
            {"role": "assistant", "content": "I agree, we should take turns."},
            {"role": "user", "content": "Great, can we also keep answers short?"},
            {"role": "assistant", "content": "Yes, short answers are usually best."},
        ]

    def shutdown(self):
        pass

    def generate_prompt_with_history(self, text, history, max_length=2048):
        prompt = []
        prompt.extend(self.initial_prompt)

        for dialogue in history:
            prompt.append({"role": "user", "content": dialogue[0]})
            prompt.append({"role": "assistant", "content": dialogue[1]})

        prompt.append({"role": "user", "content": text})
        tokens = self.tokenizer.apply_chat_template(prompt, return_tensors="np")

        if len(tokens) <= max_length:
            return tokens
        else:
            return None

    def greedy_search(
        self,
        input_ids,
        tokenizer,
        max_length: int,
        token_printing_step: int = 4,
    ):
        generated_tokens = []

        tokens = np.asarray(input_ids, dtype=np.int64)
        tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, self.binding_device)

        seq_len = tokens.shape()[1]
        past_seq_len = 0

        # Bind the main model's inputs/outputs
        self.llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))
        self.llm_io_binding.bind_output("logits", "cpu")

        for i in range(max_length):
            if i == 0:
                position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
                self.llm_io_binding.bind_cpu_input("position_ids", position_ids)
            else:
                position_ids_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))
                self.llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

            seqlens_k = np.array(past_seq_len, dtype=np.int32, ndmin=1)
            self.llm_io_binding.bind_cpu_input("seqlens_k", seqlens_k)

            # Bind the inputs/outputs of the LLM
            self.llm_io_binding.bind_ortvalue_input("tokens", tokens)
            self.llm_io_binding.bind_ortvalue_input("tokens_increment", self.tokens_increment)

            for layer_idx in range(self.num_layers):
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", self.v_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", self.v_caches[layer_idx])

            # Run the LLM
            self.llm_session.run_with_iobinding(self.llm_io_binding)
            self.llm_io_binding.synchronize_outputs()

            # Decide the next token using your preferred sampling strategy.
            logits = self.llm_io_binding.get_outputs()[0].numpy()[:, -1, :]
            next_token = np.argmax(logits, axis=-1, keepdims=True)
            generated_tokens.append(next_token.item())

            # Set the token for the next iteration
            self.tokens_increment = onnxruntime.OrtValue.ortvalue_from_numpy(next_token, self.binding_device)

            if i % token_printing_step == 0:
                yield tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if generated_tokens[-1] == tokenizer.eos_token_id:
                yield tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return

            self.llm_io_binding.bind_ortvalue_input("tokens_increment", self.tokens_increment)

            if i == 0:
                logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    (1, 1, self.tokenizer.vocab_size), self.data_type, self.binding_device
                )
                self.llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
                self.llm_io_binding.bind_ortvalue_output("logits", logits)

            past_seq_len = seq_len
            seq_len += 1

    def predict(
        self,
        text,
        chatbot,
        history,
        max_length_tokens,
        max_context_length_tokens,
        token_printing_step,
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return

        inputs = self.generate_prompt_with_history(text, history, max_length=max_context_length_tokens)

        if inputs is None:
            yield chatbot, history, "Input too long."
            return

        input_ids = inputs[:, -max_context_length_tokens:]

        # global total_count
        self.total_count += 1
        print(self.total_count)

        x = input_ids

        for x in self.greedy_search(
            input_ids,
            self.tokenizer,
            max_length=max_length_tokens,
            token_printing_step=token_printing_step,
        ):
            sentence = x

            if is_stop_word_or_prefix(sentence, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in sentence:
                    sentence = sentence[: sentence.index("[|Human|]")].strip()
                if "[|AI|]" in sentence:
                    sentence = sentence[: sentence.index("[|AI|]")].strip()
                sentence = sentence.strip()
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [[text, convert_to_markdown(sentence)]], [
                    *history,
                    [text, sentence],
                ]
                yield a, b, "Generating..."

            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except Exception as e:
                    print(type(e).__name__, e)

        del input_ids
        gc.collect()

        try:
            yield a, b, "Generate: Success"
        except Exception as e:
            print(type(e).__name__, e)

        return

    def retry(self, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step):
        if len(history) == 0:
            yield chatbot, history, "Empty context"
            return
        chatbot.pop()
        inputs = history.pop()[0]
        yield from self.predict(
            inputs,
            chatbot,
            history,
            max_length_tokens,
            max_context_length_tokens,
            token_printing_step,
        )