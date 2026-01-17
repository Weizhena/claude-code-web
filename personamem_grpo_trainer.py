"""
PersonaMem GRPO Trainer

Train inference model on PersonaMem-v2 dataset using GRPO algorithm.
- Rollout: 16 paths
- Reward: Answer matching (multiple choice)
- Output format: Answer + Evidence snippets with reasoning
- Base model: Qwen2.5-4B-Instruct

Usage:
    python -m mem_alpha.personamem_grpo_trainer --data-dir ./data/personamem
"""

import json
import re
import random
import math
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import sys


def _configure_unbuffered_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


_configure_unbuffered_logging()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

deepspeed = None


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PersonaMemSample:
    """A single training sample"""
    sample_id: str
    context: str  # Retrieved context (relevant + irrelevant mixed)
    question: str  # Parsed question text
    choices: List[str]  # ["A. xxx", "B. xxx", ...]
    correct_label: str  # "A", "B", "C", or "D"
    correct_answer_text: str  # Full text of correct answer
    incorrect_answers: List[str]  # Full text of incorrect answers
    num_relevant: int
    num_irrelevant: int
    relevant_indices: List[int]


@dataclass
class ModelOutput:
    """Parsed model output"""
    answer: str  # A/B/C/D
    evidence_snippets: List[str]  # List of evidence text
    reasoning: List[str]  # List of reasoning for each evidence
    raw_output: str  # Original model output


@dataclass
class RolloutResult:
    """Result from a single rollout"""
    sample_id: str
    prompt: str
    output: ModelOutput
    reward: float  # Total reward
    is_correct: bool
    log_prob: float
    reward_breakdown: Any = None  # Detailed reward breakdown (RewardBreakdown)


@dataclass
class GRPOBatch:
    """Batch of rollouts for GRPO training"""
    rollouts: List[RolloutResult]
    advantages: List[float]
    mean_reward: float
    accuracy: float


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PersonaMemGRPOConfig:
    """Configuration for PersonaMem GRPO training"""
    # Data
    train_path: str = "./data/personamem/train_2irrel.jsonl"
    val_path: str = "./data/personamem/val_2irrel.jsonl"

    # Model
    model_name: str = "Qwen/Qwen2.5-4B-Instruct"
    lora_adapter_path: Optional[str] = None
    deepspeed_config: Optional[str] = None
    device: str = "cuda:7"
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    max_seq_length: int = 8192
    use_cache: bool = False

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # GRPO
    rollout_n: int = 4  # Number of rollout paths
    clip_epsilon: float = 0.2
    kl_coef: float = 0.0  # Disabled as per Mem-alpha paper

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 2  # Number of samples per batch (each generates rollout_n responses)
    gradient_accumulation_steps: int = 4
    num_epochs: int = 2
    max_steps: int = 1500
    warmup_steps: int = 20
    seed: int = 42

    # Reward
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    evidence_bonus: float = 0.1  # Bonus for citing relevant evidence

    # Output
    output_dir: str = "./output/personamem_grpo"
    save_every: int = 10
    eval_every: int = 50
    log_every: int = 5
    run_final_eval: bool = True


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent assistant that answers multiple-choice questions based on retrieved context.

Your task:
1. Read the retrieved context carefully
2. Identify relevant information for answering the question
3. Select the correct answer from the choices
4. Provide evidence snippets from the context that support your answer

Output format (you MUST follow this format exactly):
Answer: [A/B/C/D]

Evidence and Reasoning:
Evidence 1: [Copy the relevant text snippet from context]
Reasoning 1: [Explain why this evidence supports your answer]

Evidence 2: [Copy another relevant text snippet if applicable]
Reasoning 2: [Explain why this evidence supports your answer]

(Continue for all relevant evidence snippets)"""


USER_PROMPT_TEMPLATE = """Retrieved Context:
{context}

Question: {question}

Choices:
{choices}

Based on the retrieved context, select the correct answer and provide evidence with reasoning."""


# ============================================================================
# Output Parser
# ============================================================================

class OutputParser:
    """Parse model output into structured format"""

    @staticmethod
    def parse(raw_output: str) -> ModelOutput:
        """
        Parse model output to extract answer and evidence

        Expected format:
        Answer: B

        Evidence and Reasoning:
        Evidence 1: [text]
        Reasoning 1: [explanation]
        Evidence 2: [text]
        Reasoning 2: [explanation]
        """
        answer = ""
        evidence_snippets = []
        reasoning = []

        # Extract answer
        answer_match = re.search(r'Answer:\s*([A-D])', raw_output, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
        else:
            # Try to find standalone letter at start
            first_line = raw_output.strip().split('\n')[0]
            letter_match = re.match(r'^([A-D])[.:\s]', first_line, re.IGNORECASE)
            if letter_match:
                answer = letter_match.group(1).upper()

        # Extract evidence snippets
        evidence_pattern = r'Evidence\s*\d*:\s*(.+?)(?=(?:Reasoning\s*\d*:|Evidence\s*\d*:|$))'
        evidence_matches = re.findall(evidence_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        for match in evidence_matches:
            cleaned = match.strip()
            if cleaned and len(cleaned) > 10:
                evidence_snippets.append(cleaned)

        # Extract reasoning
        reasoning_pattern = r'Reasoning\s*\d*:\s*(.+?)(?=(?:Evidence\s*\d*:|Reasoning\s*\d*:|$))'
        reasoning_matches = re.findall(reasoning_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        for match in reasoning_matches:
            cleaned = match.strip()
            if cleaned:
                reasoning.append(cleaned)

        return ModelOutput(
            answer=answer,
            evidence_snippets=evidence_snippets,
            reasoning=reasoning,
            raw_output=raw_output
        )


# ============================================================================
# Reward Calculator (Multi-Dimensional)
# ============================================================================
# Import from personamem_rewards for detailed reward calculation
try:
    from .personamem_rewards import (
        MultiDimensionalRewardCalculator,
        RewardConfig,
        RewardBreakdown,
        OutputParser as RewardOutputParser
    )
except ImportError:  # Fallback for running as a script
    from personamem_rewards import (
        MultiDimensionalRewardCalculator,
        RewardConfig,
        RewardBreakdown,
        OutputParser as RewardOutputParser
    )


class PersonaMemRewardCalculator:
    """
    Multi-dimensional reward calculator for PersonaMem task

    Reward components:
    - r_format (0.0 - 0.2): Output format correctness
    - r_answer (0.0 or 1.0): Answer correctness
    - r_evidence (-0.1 - 0.3): Evidence citation quality

    Total: R = r_format + r_answer + r_evidence
    Range: -0.1 to 1.5
    """

    def __init__(
        self,
        config: RewardConfig = None,
        correct_reward: Optional[float] = None,
        incorrect_reward: Optional[float] = None,
        evidence_bonus: Optional[float] = None
    ):
        reward_config = config or RewardConfig()
        if correct_reward is not None:
            reward_config.answer_correct_reward = correct_reward
        if incorrect_reward is not None:
            reward_config.answer_incorrect_reward = incorrect_reward
        if evidence_bonus is not None:
            reward_config.evidence_relevant_reward = evidence_bonus
        self.calculator = MultiDimensionalRewardCalculator(reward_config)

    def calculate(self,
                  output: ModelOutput,
                  correct_label: str,
                  context: str,
                  relevant_indices: List[int] = None) -> Tuple[float, bool, RewardBreakdown]:
        """
        Calculate multi-dimensional reward

        Args:
            output: Parsed model output
            correct_label: Correct answer label (A/B/C/D)
            context: Retrieved context string
            relevant_indices: List of relevant item indices (1-indexed)

        Returns:
            (total_reward, is_correct, breakdown)
        """
        relevant_indices = relevant_indices or []

        total, breakdown = self.calculator.calculate(
            raw_output=output.raw_output,
            correct_label=correct_label,
            context=context,
            relevant_indices=relevant_indices
        )

        return total, breakdown.answer_correct, breakdown


# ============================================================================
# Data Loader
# ============================================================================

class PersonaMemDataset(Dataset):
    """Dataset for PersonaMem training"""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.samples: List[PersonaMemSample] = []
        self._load_data(data_path, max_samples)

    def _parse_question(self, question_field: Any) -> str:
        """Parse question field which may be a dict-like string or plain text"""
        if not question_field:
            return ""

        question_str = str(question_field)

        # Try to parse as dict-like string: {'role': 'user', 'content': '...'}
        if question_str.startswith('{') and 'content' in question_str:
            try:
                # Handle single quotes by replacing with double quotes
                json_str = question_str.replace("'", '"')
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'content' in parsed:
                    return parsed['content']
            except json.JSONDecodeError:
                pass

            # Regex fallback for content extraction
            match = re.search(r"'content':\s*'([^']*)'", question_str)
            if match:
                return match.group(1)
            match = re.search(r'"content":\s*"([^"]*)"', question_str)
            if match:
                return match.group(1)

        return question_str

    def _load_data(self, data_path: str, max_samples: Optional[int] = None):
        """Load data from JSONL file"""
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                item = json.loads(line)

                # Parse question
                question = self._parse_question(item.get('question', ''))

                # Get answers
                correct_answer_text = item.get('correct_answer', '')
                incorrect_answers = item.get('incorrect_answers', [])

                # Build choices: correct + incorrect, then shuffle
                all_answers = [correct_answer_text] + incorrect_answers
                random.shuffle(all_answers)

                # Find correct label after shuffle
                correct_idx = all_answers.index(correct_answer_text)
                labels = ['A', 'B', 'C', 'D']
                correct_label = labels[correct_idx] if correct_idx < len(labels) else 'A'

                # Build choice strings
                choices = []
                for j, ans in enumerate(all_answers[:4]):  # Max 4 choices
                    label = labels[j]
                    # Truncate long answers for display
                    ans_display = ans[:200] + "..." if len(ans) > 200 else ans
                    choices.append(f"{label}. {ans_display}")

                sample = PersonaMemSample(
                    sample_id=item['sample_id'],
                    context=item.get('retrieved_context', ''),
                    question=question,
                    choices=choices,
                    correct_label=correct_label,
                    correct_answer_text=correct_answer_text,
                    incorrect_answers=incorrect_answers,
                    num_relevant=item.get('num_relevant', 0),
                    num_irrelevant=item.get('num_irrelevant', 0),
                    relevant_indices=item.get('relevant_indices', [])
                )
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PersonaMemSample:
        return self.samples[idx]


# ============================================================================
# Policy Model
# ============================================================================

class PolicyModel(ABC):
    """Abstract base class for policy model"""

    @abstractmethod
    def generate(self, prompts: List[str], n_samples: int = 1) -> List[List[str]]:
        """Generate responses for prompts"""
        pass

    @abstractmethod
    def compute_log_probs(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute log probabilities of responses"""
        pass

    @abstractmethod
    def update(self,
               prompts: List[str],
               responses: List[str],
               advantages: List[float],
               old_log_probs: List[float],
               clip_epsilon: float) -> Dict[str, float]:
        """Update policy with GRPO objective"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        pass


class QwenPolicy(PolicyModel):
    """Policy model using Qwen2.5-4B-Instruct"""

    def __init__(self, config: PersonaMemGRPOConfig):
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("QwenPolicy requires torch and transformers")

        if config.lora_adapter_path and not HAS_PEFT:
            raise ImportError("LoRA training requires peft")

        self.config = config
        self.engine = None
        self.device = config.device
        self._use_deepspeed = bool(config.deepspeed_config)
        self.deepspeed = None

        if self._use_deepspeed:
            try:
                import deepspeed as ds
            except Exception as exc:
                raise ImportError(f"DeepSpeed import failed: {exc}") from exc

            self.deepspeed = ds
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                self.deepspeed.init_distributed()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)

        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        quantization_config = None
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

        device_map = None if self._use_deepspeed else config.device
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=None if quantization_config else dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
        print("Base model loaded", flush=True)

        if config.lora_adapter_path:
            print(f"Loading LoRA adapter: {config.lora_adapter_path}", flush=True)
            self.model = PeftModel.from_pretrained(
                self.model,
                config.lora_adapter_path,
                is_trainable=True
            )
            print("LoRA adapter loaded", flush=True)
            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()

        if hasattr(self.model, "config"):
            self.model.config.use_cache = config.use_cache

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup optimizer (trainable parameters only)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate)

        if self._use_deepspeed:
            deepspeed_config = self._load_deepspeed_config()
            self.engine, self.optimizer, _, _ = self.deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=deepspeed_config,
                model_parameters=trainable_params
            )
            self.model = self.engine.module
            self.device = self.engine.device

        # Reference model for KL (optional)
        self.ref_model = None

        print(f"Model loaded on {self.device}")

    def _is_main_process(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _load_deepspeed_config(self) -> Dict[str, Any]:
        if not self.config.deepspeed_config:
            raise ValueError("DeepSpeed config path is required")

        config_path = Path(self.config.deepspeed_config)
        if not config_path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            ds_config = json.load(handle)

        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

        ds_config["train_micro_batch_size_per_gpu"] = self.config.batch_size
        ds_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps
        ds_config["train_batch_size"] = (
            self.config.batch_size * self.config.gradient_accumulation_steps * world_size
        )

        ds_config.setdefault("bf16", {})
        ds_config.setdefault("fp16", {})
        ds_config["bf16"]["enabled"] = self.config.dtype == "bfloat16"
        ds_config["fp16"]["enabled"] = self.config.dtype == "float16"

        return ds_config

    def _build_chat_input(self, prompt: str) -> str:
        """Build chat format input"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, prompts: List[str], n_samples: int = 1) -> List[List[str]]:
        """Generate n_samples responses for each prompt"""
        self.model.eval()
        all_responses = []

        with torch.no_grad():
            for prompt in prompts:
                chat_input = self._build_chat_input(prompt)
                inputs = self.tokenizer(
                    chat_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length - self.config.max_new_tokens
                ).to(self.device)

                prompt_responses = []
                for sample_idx in range(1, n_samples + 1):
                    print(f"    [Generate] Prompt {len(all_responses) + 1}/{len(prompts)} sample {sample_idx}/{n_samples}", flush=True)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=self.config.use_cache
                    )

                    # Decode only the generated part
                    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    response = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    prompt_responses.append(response)

                all_responses.append(prompt_responses)

        return all_responses

    def compute_log_probs(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute log probabilities of responses given prompts"""
        self.model.eval()
        log_probs = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                chat_input = self._build_chat_input(prompt)
                full_text = chat_input + response

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length
                ).to(self.device)

                prompt_inputs = self.tokenizer(
                    chat_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length
                )
                prompt_len = prompt_inputs['input_ids'].shape[1]

                # Forward pass
                outputs = self.model(**inputs, use_cache=self.config.use_cache)
                logits = outputs.logits

                # Compute log probs for response tokens only
                input_ids = inputs['input_ids'][0]
                seq_len = input_ids.shape[0]
                response_len = seq_len - prompt_len
                start = prompt_len
                end = seq_len - 1
 
                if end <= start:
                    log_probs.append(0.0)
                    continue
 
                positions = torch.arange(start, end, device=logits.device)
                target_ids = input_ids[positions + 1]
                logits_slice = logits[0, positions, :]
                log_probs_slice = F.log_softmax(logits_slice, dim=-1)
                token_log_probs = log_probs_slice.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
 
                total_log_prob = token_log_probs.sum().item()
                if response_len > 0:
                    total_log_prob = total_log_prob / response_len
 
                log_probs.append(total_log_prob)


        return log_probs

    def update(self,
               prompts: List[str],
               responses: List[str],
               advantages: List[float],
               old_log_probs: List[float],
               clip_epsilon: float) -> Dict[str, float]:
        """
        Update policy using GRPO objective

        Loss = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
        """
        if self.engine:
            self.engine.train()
            self.engine.zero_grad()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        total_ratio = 0.0
        num_clipped = 0
        total_loss_value = 0.0
        mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0

        for prompt, response, advantage, old_log_prob in zip(
            prompts, responses, advantages, old_log_probs
        ):
            chat_input = self._build_chat_input(prompt)
            full_text = chat_input + response

            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.device)

            prompt_inputs = self.tokenizer(
                chat_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            )
            prompt_len = prompt_inputs['input_ids'].shape[1]

            # Forward pass
            outputs = self.model(
                **inputs,
                use_cache=self.config.use_cache
            )
            logits = outputs.logits

            # Compute new log prob for response tokens only
            input_ids = inputs['input_ids'][0]
            seq_len = input_ids.shape[0]
            response_len = seq_len - prompt_len
            start = prompt_len
            end = seq_len - 1

            if end <= start:
                new_log_prob = torch.tensor(0.0, device=self.device)
            else:
                positions = torch.arange(start, end, device=logits.device)
                target_ids = input_ids[positions + 1]
                logits_slice = logits[0, positions, :]
                log_probs_slice = F.log_softmax(logits_slice, dim=-1)
                token_log_probs = log_probs_slice.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                new_log_prob = token_log_probs.sum()
                if response_len > 0:
                    new_log_prob = new_log_prob / response_len

            # Compute ratio
            ratio = torch.exp(new_log_prob - old_log_prob)

            # Clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

            # Loss term
            advantage_tensor = torch.tensor(advantage, device=self.device)
            loss_unclipped = -ratio * advantage_tensor
            loss_clipped = -clipped_ratio * advantage_tensor
            loss = torch.max(loss_unclipped, loss_clipped)

            scaled_loss = loss / len(prompts)
            if self.engine:
                self.engine.backward(scaled_loss)
            else:
                scaled_loss.backward()

            total_loss_value += loss.item()
            total_ratio += ratio.item()
            if ratio.item() != clipped_ratio.item():
                num_clipped += 1

        if self.engine:
            self.engine.step()
        else:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update
            self.optimizer.step()

        avg_loss_value = total_loss_value / len(prompts)
        return {
            "loss": avg_loss_value,
            "mean_ratio": total_ratio / len(prompts),
            "clip_fraction": num_clipped / len(prompts),
            "mean_advantage": mean_advantage
        }

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.engine:
            self.engine.save_checkpoint(path)
            if self._is_main_process():
                self.tokenizer.save_pretrained(path)
                print(f"Model saved to {path}")
            return

        if self._is_main_process():
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model saved to {path}")


class MockQwenPolicy(PolicyModel):
    """Mock policy for testing without GPU"""

    def __init__(self, config: PersonaMemGRPOConfig):
        self.config = config
        self.update_count = 0

    def generate(self, prompts: List[str], n_samples: int = 1) -> List[List[str]]:
        """Generate mock responses"""
        all_responses = []
        for prompt in prompts:
            responses = []
            for _ in range(n_samples):
                answer = random.choice(['A', 'B', 'C', 'D'])
                response = f"""Answer: {answer}

Evidence and Reasoning:
Evidence 1: Based on the conversation context provided.
Reasoning 1: This answer is most consistent with the information given.

Evidence 2: The context mentions relevant details supporting this choice.
Reasoning 2: This aligns with the facts presented in the retrieved information."""
                responses.append(response)
            all_responses.append(responses)
        return all_responses

    def compute_log_probs(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Return mock log probs"""
        return [-1.0 + random.random() * 0.5 for _ in prompts]

    def update(self,
               prompts: List[str],
               responses: List[str],
               advantages: List[float],
               old_log_probs: List[float],
               clip_epsilon: float) -> Dict[str, float]:
        """Mock update"""
        self.update_count += 1
        return {
            "loss": random.random() * 0.5,
            "mean_ratio": 1.0 + random.random() * 0.1,
            "clip_fraction": random.random() * 0.2,
            "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0
        }

    def save(self, path: str) -> None:
        """Mock save"""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "mock_checkpoint.json", 'w') as f:
            json.dump({"update_count": self.update_count}, f)


# ============================================================================
# GRPO Trainer
# ============================================================================

class PersonaMemGRPOTrainer:
    """
    GRPO Trainer for PersonaMem Inference Model

    Implements Group Relative Policy Optimization with:
    - Rollout sampling (16 paths per sample)
    - Group-relative advantage normalization
    - Answer matching reward
    """

    def __init__(self,
                 policy: PolicyModel,
                 config: PersonaMemGRPOConfig):
        self.policy = policy
        self.config = config
        self.reward_calc = PersonaMemRewardCalculator(
            correct_reward=config.correct_reward,
            incorrect_reward=config.incorrect_reward,
            evidence_bonus=config.evidence_bonus
        )
        self.parser = OutputParser()

        # Training state
        self.current_step = 0
        self.training_history: List[Dict] = []

        # Output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _format_prompt(self, sample: PersonaMemSample) -> str:
        """Format sample into prompt"""
        choices_str = "\n".join(sample.choices)
        return USER_PROMPT_TEMPLATE.format(
            context=sample.context,
            question=sample.question,
            choices=choices_str
        )

    def collect_rollouts(self, samples: List[PersonaMemSample]) -> List[RolloutResult]:
        """
        Collect rollouts for a batch of samples

        For each sample, generate rollout_n responses and compute rewards
        """
        rollouts = []

        # Format prompts
        prompts = [self._format_prompt(s) for s in samples]

        print(f"  [Rollout] Generating {self.config.rollout_n} responses per prompt...", flush=True)
        # Generate responses (n_samples per prompt)
        all_responses = self.policy.generate(prompts, n_samples=self.config.rollout_n)
        print("  [Rollout] Generation complete, computing rewards...", flush=True)

        tokenizer = getattr(self.policy, "tokenizer", None)
        for sample_idx, responses in enumerate(all_responses, start=1):
            print(f"  [Rollout] Sample {sample_idx}/{len(all_responses)} responses: {len(responses)}", flush=True)
            if tokenizer is not None:
                try:
                    lengths = [len(tokenizer(r).input_ids) for r in responses]
                    if lengths:
                        mean_len = sum(lengths) / len(lengths)
                        print(
                            "  [Rollout] Response token lengths "
                            f"(sample {sample_idx}): min={min(lengths)}, "
                            f"max={max(lengths)}, mean={mean_len:.1f}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"  [Rollout] Token length calc failed: {exc}", flush=True)

        # Process each sample's responses
        for sample, prompt, responses in zip(samples, prompts, all_responses):
            # Compute log probs for all responses
            print("  [Rollout] Computing log_probs...", flush=True)
            log_probs = self.policy.compute_log_probs(
                [prompt] * len(responses),
                responses
            )
            print("  [Rollout] Log_probs complete", flush=True)

            for response_idx, (response, log_prob) in enumerate(zip(responses, log_probs), start=1):
                print(f"  [Reward] Parsing response {response_idx}/{len(responses)}", flush=True)
                # Parse output
                parsed = self.parser.parse(response)

                # Calculate multi-dimensional reward
                reward, is_correct, breakdown = self.reward_calc.calculate(
                    parsed,
                    sample.correct_label,
                    sample.context,
                    sample.relevant_indices  # Pass relevant indices for evidence reward
                )
                print(f"  [Reward] Done response {response_idx}/{len(responses)}", flush=True)

                rollout = RolloutResult(
                    sample_id=sample.sample_id,
                    prompt=prompt,
                    output=parsed,
                    reward=reward,
                    is_correct=is_correct,
                    log_prob=log_prob,
                    reward_breakdown=breakdown  # Store detailed breakdown
                )
                rollouts.append(rollout)

        return rollouts

    def compute_advantages(self, rollouts: List[RolloutResult]) -> List[float]:
        """
        Compute group-relative advantages for GRPO

        A = (r - mean) / (std + eps)
        """
        if not rollouts:
            return []

        rewards = [r.reward for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance) if variance > 0 else 1.0
        epsilon = 1e-8

        advantages = [(r - mean_reward) / (std_reward + epsilon) for r in rewards]
        return advantages

    def train_step(self, samples: List[PersonaMemSample]) -> Dict[str, float]:
        """Perform one training step"""
        # Collect rollouts
        rollouts = self.collect_rollouts(samples)

        # Compute advantages
        advantages = self.compute_advantages(rollouts)

        # Prepare update data
        prompts = [r.prompt for r in rollouts]
        responses = [r.output.raw_output for r in rollouts]
        old_log_probs = [r.log_prob for r in rollouts]

        # Update policy
        print("  [Update] Updating policy...", flush=True)
        update_metrics = self.policy.update(
            prompts=prompts,
            responses=responses,
            advantages=advantages,
            old_log_probs=old_log_probs,
            clip_epsilon=self.config.clip_epsilon
        )
        print("  [Update] Update complete", flush=True)

        # Compute metrics
        mean_reward = sum(r.reward for r in rollouts) / len(rollouts)
        accuracy = sum(1 for r in rollouts if r.is_correct) / len(rollouts)

        metrics = {
            "step": self.current_step,
            "loss": update_metrics["loss"],
            "mean_reward": mean_reward,
            "accuracy": accuracy,
            "mean_ratio": update_metrics["mean_ratio"],
            "clip_fraction": update_metrics["clip_fraction"],
            "num_rollouts": len(rollouts)
        }

        self.training_history.append(metrics)
        self.current_step += 1

        return metrics

    def evaluate(self, samples: List[PersonaMemSample]) -> Dict[str, float]:
        """Evaluate on validation samples with detailed metrics"""
        # Generate single response per sample for evaluation
        prompts = [self._format_prompt(s) for s in samples]
        all_responses = self.policy.generate(prompts, n_samples=1)

        correct = 0
        total_reward = 0.0
        total_r_format = 0.0
        total_r_answer = 0.0
        total_r_evidence = 0.0

        for sample, responses in zip(samples, all_responses):
            response = responses[0]
            parsed = self.parser.parse(response)
            reward, is_correct, breakdown = self.reward_calc.calculate(
                parsed,
                sample.correct_label,
                sample.context,
                sample.relevant_indices
            )
            if is_correct:
                correct += 1
            total_reward += reward
            total_r_format += breakdown.r_format
            total_r_answer += breakdown.r_answer
            total_r_evidence += breakdown.r_evidence

        n = len(samples)
        return {
            "accuracy": correct / n,
            "mean_reward": total_reward / n,
            "mean_r_format": total_r_format / n,
            "mean_r_answer": total_r_answer / n,
            "mean_r_evidence": total_r_evidence / n,
            "num_samples": n
        }

    def train(self,
              train_dataset: PersonaMemDataset,
              val_dataset: Optional[PersonaMemDataset] = None) -> List[Dict]:
        """
        Full training loop

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset

        Returns:
            Training history
        """
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1
        is_main = rank == 0

        if is_main:
            print("=" * 60)
            print("PersonaMem GRPO Training")
            print("=" * 60)
            print(f"Config:")
            print(f"  Model: {self.config.model_name}")
            print(f"  Rollout N: {self.config.rollout_n}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  Max steps: {self.config.max_steps}")
            print(f"  Device: {self.config.device}")
            print(f"  DeepSpeed config: {self.config.deepspeed_config}")
            print(f"  Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"  Validation samples: {len(val_dataset)}")
            if world_size > 1:
                print(f"  World size: {world_size}")
            print("=" * 60)

        full_samples = list(train_dataset.samples)

        for epoch in range(self.config.num_epochs):
            if is_main:
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                print("-" * 40)

            # Shuffle full training data deterministically across ranks
            rng = random.Random(self.config.seed + epoch)
            rng.shuffle(full_samples)

            if is_distributed:
                train_samples = full_samples[rank::world_size]
                print(f"[Rank {rank}] shard size: {len(train_samples)}", flush=True)
            else:
                train_samples = full_samples

            # Process batches
            for batch_start in range(0, len(train_samples), self.config.batch_size):
                if self.current_step >= self.config.max_steps:
                    if is_main:
                        print(f"\nReached max steps ({self.config.max_steps})")
                    break

                batch_end = min(batch_start + self.config.batch_size, len(train_samples))
                batch = train_samples[batch_start:batch_end]

                if is_main:
                    print(f"\n[Step {self.current_step}] Collecting rollouts for batch size {len(batch)}")
                # Training step
                metrics = self.train_step(batch)
                if is_main:
                    print(f"[Step {self.current_step}] Update done")

                # Logging
                if is_main and self.current_step % self.config.log_every == 0:
                    print(f"  Step {metrics['step']:4d}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.4f}, "
                          f"acc={metrics['accuracy']:.4f}, "
                          f"ratio={metrics['mean_ratio']:.4f}")

                # Evaluation
                if is_main and val_dataset and self.current_step % self.config.eval_every == 0:
                    val_metrics = self.evaluate(val_dataset.samples[:100])
                    print(f"  [Val] accuracy={val_metrics['accuracy']:.4f}, "
                          f"reward={val_metrics['mean_reward']:.4f}")

                # Save checkpoint
                if is_main and self.current_step % self.config.save_every == 0:
                    self.save_checkpoint()

            if self.current_step >= self.config.max_steps:
                break

        # Final evaluation
        if is_main and val_dataset and self.config.run_final_eval:
            print("\n" + "=" * 60)
            print("Final Evaluation")
            print("=" * 60)
            final_metrics = self.evaluate(val_dataset.samples)
            print(f"Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"Mean Reward: {final_metrics['mean_reward']:.4f}")

        # Save final checkpoint
        self.save_checkpoint(final=True)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total steps: {self.current_step}")
        if self.training_history:
            print(f"Final accuracy: {self.training_history[-1]['accuracy']:.4f}")

        return self.training_history

    def save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint"""
        suffix = "final" if final else f"step{self.current_step}"

        # Save model
        model_path = Path(self.config.output_dir) / f"model_{suffix}"
        self.policy.save(str(model_path))

        # Save training state
        state = {
            "step": self.current_step,
            "config": {
                "model_name": self.config.model_name,
                "rollout_n": self.config.rollout_n,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "clip_epsilon": self.config.clip_epsilon
            },
            "history": self.training_history[-100:]  # Last 100 steps
        }

        state_path = Path(self.config.output_dir) / f"training_state_{suffix}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"  Checkpoint saved: {model_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="PersonaMem GRPO Training")
    parser.add_argument("--data-dir", type=str, default="./data/personamem",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./output/personamem_grpo",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-4B-Instruct",
                        help="Model name")
    parser.add_argument("--lora-adapter", type=str, default="./output/personamem_sft_lora/lora_adapter",
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--deepspeed-config", type=str, default="./deepspeed_grpo.json",
                        help="DeepSpeed config path (optional)")
    parser.add_argument("--device", type=str, default="cuda:7",
                        help="Device to run on (non-DeepSpeed)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit to save memory")
    parser.add_argument("--rollout-n", type=int, default=4,
                        help="Number of rollout paths")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum new tokens during generation")
    parser.add_argument("--use-cache", action="store_true",
                        help="Enable KV cache during generation")
    parser.add_argument("--no-final-eval", action="store_true",
                        help="Skip final evaluation phase")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock policy (for testing without GPU)")

    args = parser.parse_args()

    deepspeed_config = args.deepspeed_config
    if isinstance(deepspeed_config, str) and deepspeed_config.lower() in {"", "none"}:
        deepspeed_config = None

    # Create config
    config = PersonaMemGRPOConfig(
        train_path=f"{args.data_dir}/train_2irrel.jsonl",
        val_path=f"{args.data_dir}/val_2irrel.jsonl",
        model_name=args.model,
        lora_adapter_path=args.lora_adapter,
        deepspeed_config=deepspeed_config,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        rollout_n=args.rollout_n,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_cache=args.use_cache,
        output_dir=args.output_dir
    )

    # Load data
    print(f"PID: {os.getpid()}")
    print("Loading datasets...")
    train_dataset = PersonaMemDataset(config.train_path, max_samples=args.max_samples)
    val_dataset = PersonaMemDataset(config.val_path, max_samples=args.max_samples // 5 if args.max_samples else None)

    # Create policy
    if args.mock:
        print("Using mock policy (no GPU)")
        policy = MockQwenPolicy(config)
    else:
        print("Loading real policy model...")
        policy = QwenPolicy(config)

    # Create trainer
    print("Initializing trainer...", flush=True)
    trainer = PersonaMemGRPOTrainer(policy=policy, config=config)
    print("Trainer ready", flush=True)

    # Train
    print("Starting training loop...", flush=True)
    history = trainer.train(train_dataset, val_dataset)

    return history


if __name__ == "__main__":
    main()
