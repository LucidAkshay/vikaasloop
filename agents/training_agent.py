# agents/training_agent.py
# AGPL v3 - VikaasLoop
#
# TrainingAgent: fine-tunes a base model with a QLoRA adapter using TRL SFTTrainer.
# One instance per training run. Not shared between iterations.

import os
import time
import asyncio
import re
import threading
import logging
import bitsandbytes  # Required for Windows environment variable setup and 4-bit optimisers
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

from utils.websocket_manager import ws_manager
from utils.formatter import format_training_pair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Lock
# ---------------------------------------------------------------------------
TRAINING_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_MODELS = [
    "microsoft/phi-2",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
]

ALLOWED_DATA_DIR = os.path.realpath("data")
MAX_TOKENIZER_CACHE_SIZE = 2


# ---------------------------------------------------------------------------
# Loss streaming callback
# ---------------------------------------------------------------------------

class LossStreamingCallback(TrainerCallback):
    """Streams training loss to the frontend via the shared WebSocket manager."""

    def __init__(self, run_id: str, loop: asyncio.AbstractEventLoop):
        self.run_id = run_id
        self.loop = loop

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.log_history:
            return
        last = state.log_history[-1]
        if "loss" not in last:
            return
        try:
            if self.loop and self.loop.is_running() and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(
                        self.run_id,
                        {"type": "train_step", "step": state.global_step, "loss": last["loss"]},
                    ),
                    self.loop,
                )
        except Exception as exc:
            logger.error(f"Loss stream error for run {self.run_id}: {exc}")


# ---------------------------------------------------------------------------
# TrainingAgent
# ---------------------------------------------------------------------------

class TrainingAgent:
    """
    Fine-tunes a base model with a QLoRA adapter for one training run.
    """

    _tokenizer_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(
        self,
        base_model_name: str,
        training_data_path: str,
        run_id: str,
    ):
        if base_model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{base_model_name}' is not in the allowed list: {ALLOWED_MODELS}"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", run_id.replace("-", "_")):
            raise ValueError(
                f"Invalid run_id '{run_id}'. Only alphanumeric, hyphens, underscores."
            )

        resolved = os.path.realpath(training_data_path)
        if not (
            resolved.startswith(ALLOWED_DATA_DIR + os.sep)
            or resolved == ALLOWED_DATA_DIR
        ):
            raise ValueError(
                f"training_data_path '{training_data_path}' is outside "
                f"the allowed data directory '{ALLOWED_DATA_DIR}'."
            )
        if not os.path.isfile(resolved):
            raise ValueError(
                f"training_data_path '{training_data_path}' is not a file."
            )

        self.base_model_name = base_model_name
        self.training_data_path = resolved
        self.run_id = run_id
        self.adapter_path = os.path.join("models", run_id, "adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_target_modules(self):
        name = self.base_model_name.lower()
        if "phi-2" in name:
            return ["Wqkv", "out_proj", "fc1", "fc2"]
        if "llama" in name or "gemma" in name:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        return "all-linear"

    def _setup_tokenizer(self) -> AutoTokenizer:
        with TrainingAgent._cache_lock:
            if self.base_model_name in TrainingAgent._tokenizer_cache:
                return TrainingAgent._tokenizer_cache[self.base_model_name]

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        with TrainingAgent._cache_lock:
            if self.base_model_name in TrainingAgent._tokenizer_cache:
                return TrainingAgent._tokenizer_cache[self.base_model_name]

            if len(TrainingAgent._tokenizer_cache) >= MAX_TOKENIZER_CACHE_SIZE:
                oldest = next(iter(TrainingAgent._tokenizer_cache))
                del TrainingAgent._tokenizer_cache[oldest]
                logger.info(f"Tokenizer cache: evicted '{oldest}'.")

            TrainingAgent._tokenizer_cache[self.base_model_name] = tokenizer
            return tokenizer

    def _load_fresh_model(self) -> AutoModelForCausalLM:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Required for gradient checkpointing
        model.config.use_cache = False 
        
        if device == "cuda":
            model = prepare_model_for_kbit_training(model)
            
        return model

    def _wrap_with_lora(self, base_model) -> Any:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=self._get_target_modules(),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(base_model, lora_config)

    def _prepare_dataset(self):
        return load_dataset(
            "json",
            data_files=self.training_data_path,
            split="train",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def train(self) -> dict:
        async with TRAINING_LOCK:
            start_time = time.time()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._run_training_sync,
                start_time,
                loop,
            )

    def _run_training_sync(
        self,
        start_time: float,
        loop: asyncio.AbstractEventLoop,
    ) -> dict:
        tokenizer = self._setup_tokenizer()
        model = self._wrap_with_lora(self._load_fresh_model())
        dataset = self._prepare_dataset()

        if "text" not in dataset.column_names:
            logger.info("Formatting dataset into 'text' column.")
            dataset = dataset.map(
                lambda ex: {"text": format_training_pair(ex)}
            )

        if not dataset or len(dataset) == 0:
            raise ValueError("Training dataset is empty.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=os.path.join("models", self.run_id, "tmp"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=device == "cuda" and not bf16_supported,
            bf16=bf16_supported,
            gradient_checkpointing=True,  # Critical VRAM fix
            push_to_hub=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=training_args,
            callbacks=[LossStreamingCallback(self.run_id, loop)],
        )

        train_result = trainer.train()

        # Re-enable cache for inference after training
        model.config.use_cache = True
        
        model.save_pretrained(self.adapter_path)
        tokenizer.save_pretrained(self.adapter_path)

        return {
            "final_loss": float(train_result.training_loss),
            "training_time_seconds": time.time() - start_time,
            "adapter_path": self.adapter_path,
        }