# utils/formatter.py
# AGPL v3 - VikaasLoop
#
# Single source of truth for instruction/response formatting.
# Called by DataGenAgent, TrainingAgent, and the scoring pipeline.

from typing import Union, Dict, Any


def format_training_pair(instruction_or_dict: Union[str, Dict[str, Any]], response: Any = None) -> str:
    """
    Format an instruction and response pair into the standard training text.
    Enforces strict string conversion to prevent crashes from LLM hallucinations.

    Supports two calling conventions:
      1. format_training_pair("What is X?", "X is Y.")
      2. format_training_pair({"instruction": "What is X?", "response": "X is Y."})

    Returns the canonical training format used by SFTTrainer.
    """
    if isinstance(instruction_or_dict, dict):
        raw_instruction = instruction_or_dict.get("instruction", "")
        raw_response = instruction_or_dict.get("response", "")
    else:
        raw_instruction = instruction_or_dict
        raw_response = response

    # Zero trust data handling enforces string conversion before stripping
    safe_instruction = str(raw_instruction) if raw_instruction is not None else ""
    safe_response = str(raw_response) if raw_response is not None else ""

    return f"### Instruction:\n{safe_instruction.strip()}\n\n### Response:\n{safe_response.strip()}"