# agents/model_manager.py
# AGPL v3 - VikaasLoop
#
# ModelManager: owns model loading, caching, inference, and VRAM cleanup.
# Has zero knowledge of evaluation logic, judging, or scoring.
# EvalAgent depends on this; nothing else should need to.

import gc
import logging
import asyncio
from typing import Optional, List

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of the base model and its LoRA adapter for evaluation.

    Design principles:
    - Base model is loaded ONCE per unique model name for the lifetime of the
      Orchestrator. Subsequent calls to ensure_base_loaded with the same name
      are no-ops (KL-001 fix).
    - Adapter model reloads only when the adapter path changes between iterations.
    - All VRAM/PyTorch code lives here. No other file imports torch.
    - Thread-safe for the read path (generate_response). Model loading is
      single-threaded by design — call ensure_* before parallelising inference.

    Typical lifecycle per loop run:
        manager.ensure_adapter_loaded("microsoft/phi-2", "models/run-1/adapter")
        # -> loads base once, loads adapter
        responses = await manager.generate_batch("microsoft/phi-2", None, prompts)
        responses = await manager.generate_batch("microsoft/phi-2", "models/run-1/adapter", prompts)

        manager.ensure_adapter_loaded("microsoft/phi-2", "models/run-2/adapter")
        # -> base NOT reloaded, only adapter swapped

        manager.release()   # call once when ALL loops are done
    """

    def __init__(self):
        self._base_model = None
        self._base_tokenizer = None
        self._base_model_name: Optional[str] = None

        self._adapter_model = None
        self._adapter_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def base_model(self):
        if self._base_model is None:
            raise RuntimeError(
                "Base model is not loaded. Call ensure_base_loaded() first."
            )
        return self._base_model

    @property
    def adapter_model(self):
        if self._adapter_model is None:
            raise RuntimeError(
                "Adapter model is not loaded. Call ensure_adapter_loaded() first."
            )
        return self._adapter_model

    @property
    def tokenizer(self):
        if self._base_tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Call ensure_base_loaded() first."
            )
        return self._base_tokenizer

    @property
    def is_base_loaded(self) -> bool:
        return self._base_model is not None

    @property
    def current_base_model_name(self) -> Optional[str]:
        return self._base_model_name

    @property
    def current_adapter_path(self) -> Optional[str]:
        return self._adapter_path

    # ------------------------------------------------------------------
    # Loading — idempotent, safe to call on every iteration
    # ------------------------------------------------------------------

    def ensure_base_loaded(self, model_name: str) -> None:
        """
        Load the base model if not already loaded with this exact name.
        If the same model_name is already in memory, this is a no-op — O(1).
        If a different model_name is requested, the old one is evicted first.

        This is the authoritative caching gate.
        """
        if self._base_model_name == model_name and self._base_model is not None:
            logger.debug(f"Base model '{model_name}' already loaded — skipping.")
            return

        if self._base_model is not None:
            logger.info(
                f"Evicting base model '{self._base_model_name}' "
                f"to load '{model_name}'."
            )
            self._release_base()

        logger.info(f"Loading base model: {model_name}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left padding is required for batched inference generation
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        model.eval()

        self._base_tokenizer = tokenizer
        self._base_model = model
        self._base_model_name = model_name

        logger.info(f"Base model loaded on {device} (dtype={dtype}).")

    def ensure_adapter_loaded(self, base_model_name: str, adapter_path: str) -> None:
        """
        Load an adapter on top of the base model.
        """
        self.ensure_base_loaded(base_model_name)

        if self._adapter_path == adapter_path and self._adapter_model is not None:
            logger.debug(f"Adapter '{adapter_path}' already loaded — skipping.")
            return

        logger.info(f"Loading adapter: {adapter_path}")

        from peft import PeftModel

        self._adapter_model = None   # free previous adapter before loading new one
        self._gc()
        
        self._adapter_model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self._adapter_model.eval()
        self._adapter_path = adapter_path

        logger.info("Adapter loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate_batch(
        self, 
        base_model_name: str, 
        adapter_path: Optional[str], 
        prompts: List[str], 
        max_new_tokens: int = 256
    ) -> List[str]:
        """
        Asynchronous wrapper to generate responses for a list of prompts.
        This resolves the missing method error expected by the EvalAgent.
        """
        def _run_batch_sync():
            if adapter_path:
                self.ensure_adapter_loaded(base_model_name, adapter_path)
                target_model = self.adapter_model
            else:
                self.ensure_base_loaded(base_model_name)
                target_model = self.base_model

            import torch
            
            # Process in small chunks to avoid VRAM OOM during evaluation
            chunk_size = 4
            all_responses = []
            
            device = next(target_model.parameters()).device
            
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i:i + chunk_size]
                
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = target_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode only the newly generated tokens
                input_length = inputs["input_ids"].shape[1]
                for idx, out_id in enumerate(output_ids):
                    new_tokens = out_id[input_length:]
                    response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    all_responses.append(response_text)
                    
            return all_responses

        # Offload the heavy GPU processing to a background thread
        return await asyncio.to_thread(_run_batch_sync)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _release_base(self) -> None:
        """Release base model memory. Internal."""
        self._base_model = None
        self._base_tokenizer = None
        self._base_model_name = None
        self._gc()

    def _release_adapter(self) -> None:
        """Release adapter memory. Internal use only."""
        self._adapter_model = None
        self._adapter_path = None
        self._gc()

    def release(self) -> None:
        """
        Release ALL loaded models and free VRAM.
        Call ONCE when the entire loop run is complete.
        """
        logger.info("ModelManager: releasing all models.")
        self._adapter_model = None
        self._base_model = None
        self._base_tokenizer = None
        self._base_model_name = None
        self._adapter_path = None
        self._gc()
        logger.info("ModelManager: all models released.")

    @staticmethod
    def _gc() -> None:
        """Garbage-collect and flush CUDA cache if available."""
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()