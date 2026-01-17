"""
PersonaMem SFT Data Construction and Training (LoRA 微调版)

Constructs supervised fine-tuning data:
- Input: context + question + choices
- Output: Answer + Evidence (from relevant items) + Reasoning

Then trains the model with LoRA (Low-Rank Adaptation) to save memory.
"""

import json
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys


def _configure_unbuffered_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


_configure_unbuffered_logging()

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
HAS_TRANSFORMERS = True
HAS_PEFT = True


# ============================================================================
# Configuration (新增 LoRA 相关配置)
# ============================================================================

@dataclass
class PersonaMemSFTConfig:
    """Configuration for SFT training (LoRA 版)"""
    # Data
    train_data_path: str = "./data/personamem/train_2irrel.jsonl"
    output_sft_data_path: str = "./data/personamem/sft_train.jsonl"

    # Model
    model_name: str = "Qwen/Qwen3-4B"
    device: str = "cuda:7"
    dtype: str = "bfloat16"
    load_in_4bit: bool = True  # 可选：4-bit量化，进一步降低显存占用

    # LoRA 配置 (核心新增)
    lora_r: int = 8  # LoRA 秩，越小显存占用越低
    lora_alpha: int = 32  # 缩放因子，通常为 r 的 2-4 倍
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # Qwen3-4B 核心微调模块

    # lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", 
                    # "up_proj", "down_proj", "gate_proj"])
    # Training
    learning_rate: float = 2e-4  # LoRA 学习率可略高（比全量微调大10倍）
    batch_size: int = 1  # LoRA 显存占用低，可增大 batch_size
    gradient_accumulation_steps: int = 2
    num_epochs: int = 2
    max_seq_length: int = 4096
    warmup_ratio: float = 0.1

    # Data construction
    max_sft_samples: int = 2000
    max_evidence_per_sample: int = 2

    # Output
    output_dir: str = "./output/personamem_sft_lora"


# ============================================================================
# Prompt Templates (无改动)
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
# SFT Data Construction (无改动)
# ============================================================================

class SFTDataConstructor:
    """
    Construct SFT training data from PersonaMem dataset

    Creates (prompt, response) pairs where:
    - prompt: context + question + choices
    - response: Answer + Evidence from relevant items + Reasoning
    """

    def __init__(self, config: PersonaMemSFTConfig = None):
        self.config = config or PersonaMemSFTConfig()

    def _parse_question(self, question_field: Any) -> str:
        """Parse question from dict-like string"""
        if not question_field:
            return ""

        question_str = str(question_field)

        if question_str.startswith('{') and 'content' in question_str:
            try:
                json_str = question_str.replace("'", '"')
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'content' in parsed:
                    return parsed['content']
            except:
                pass

            match = re.search(r"'content':\s*'([^']*)'", question_str)
            if match:
                return match.group(1)

        return question_str

    def _parse_context_items(self, context: str) -> Dict[int, str]:
        """Parse context into numbered items"""
        items = {}
        pattern = r'(\d+)\.\s*\[([^\]]+)\]:\s*(.+?)(?=\n\d+\.\s*\[|\Z)'
        matches = re.findall(pattern, context, re.DOTALL)

        for num_str, role, content in matches:
            idx = int(num_str)
            items[idx] = f"[{role}]: {content.strip()}"

        return items

    def _extract_evidence_from_relevant(self,
                                        context: str,
                                        relevant_indices: List[int],
                                        max_evidence: int = 2) -> List[Tuple[str, int]]:
        """
        Extract evidence snippets from relevant context items

        Returns:
            List of (evidence_text, source_index) tuples
        """
        context_items = self._parse_context_items(context)
        evidences = []

        for idx in relevant_indices:
            if idx in context_items:
                content = context_items[idx]
                clean_content = re.sub(r'^\[[^\]]+\]:\s*', '', content)

                # Truncate if too long (keep first 300 chars for evidence)
                if len(clean_content) > 300:
                    clean_content = clean_content[:300] + "..."

                evidences.append((clean_content, idx))

                if len(evidences) >= max_evidence:
                    break

        return evidences

    def _generate_reasoning(self, evidence: str, question: str) -> str:
        """
        Generate reasoning text for evidence

        Simple template-based reasoning generation
        """
        reasoning_templates = [
            "This passage directly addresses the user's question about {topic}, providing relevant personal context and background information.",
            "The context shows the user's specific situation related to {topic}, which helps identify the most appropriate response.",
            "This information reveals the user's preferences and concerns regarding {topic}, supporting the selected answer.",
            "The passage contains key details about {topic} that align with the correct answer choice.",
        ]

        # Extract topic from question
        topic_words = question.lower().split()[:5]
        topic = " ".join(topic_words)

        template = random.choice(reasoning_templates)
        return template.format(topic=topic)

    def construct_response(self,
                           correct_label: str,
                           evidences: List[Tuple[str, int]],
                           question: str) -> str:
        """
        Construct the target response for SFT

        Format:
        Answer: B

        Evidence and Reasoning:
        Evidence 1: [evidence text]
        Reasoning 1: [reasoning]
        ...
        """
        lines = [f"Answer: {correct_label}", "", "Evidence and Reasoning:"]

        for i, (evidence, src_idx) in enumerate(evidences, 1):
            reasoning = self._generate_reasoning(evidence, question)
            lines.append(f"Evidence {i}: {evidence}")
            lines.append(f"Reasoning {i}: {reasoning}")
            lines.append("")

        return "\n".join(lines).strip()

    def construct_sft_sample(self, raw_item: Dict) -> Optional[Dict]:
        """
        Construct a single SFT sample

        Returns:
            {
                "prompt": formatted prompt,
                "response": formatted response,
                "messages": chat format for training
            }
        """
        # Parse question
        question = self._parse_question(raw_item.get('question', ''))
        if not question:
            return None

        context = raw_item.get('retrieved_context', '')
        correct_answer_text = raw_item.get('correct_answer', '')
        incorrect_answers = raw_item.get('incorrect_answers', [])
        relevant_indices = raw_item.get('relevant_indices', [])

        if not context or not correct_answer_text:
            return None

        # Build choices
        all_answers = [correct_answer_text] + incorrect_answers
        random.shuffle(all_answers)

        correct_idx = all_answers.index(correct_answer_text)
        labels = ['A', 'B', 'C', 'D']
        correct_label = labels[correct_idx] if correct_idx < len(labels) else 'A'

        choices = []
        for j, ans in enumerate(all_answers[:4]):
            label = labels[j]
            ans_display = ans[:200] + "..." if len(ans) > 200 else ans
            choices.append(f"{label}. {ans_display}")

        choices_str = "\n".join(choices)

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            choices=choices_str
        )

        # Extract evidence from relevant items
        evidences = self._extract_evidence_from_relevant(
            context,
            relevant_indices,
            self.config.max_evidence_per_sample
        )

        if not evidences:
            # Fallback: create dummy evidence
            evidences = [("Based on the context provided.", 0)]

        # Construct response
        response = self.construct_response(correct_label, evidences, question)

        # Create chat format for training
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ]

        return {
            "sample_id": raw_item.get('sample_id', ''),
            "prompt": user_prompt,
            "response": response,
            "correct_label": correct_label,
            "messages": messages,
            "metadata": {
                "relevant_indices": relevant_indices,
                "num_evidence": len(evidences)
            }
        }

    def construct_dataset(self,
                          input_path: str,
                          output_path: str,
                          max_samples: int = None) -> List[Dict]:
        """
        Construct full SFT dataset

        Args:
            input_path: Path to raw training data
            output_path: Path to save SFT data
            max_samples: Maximum samples to create

        Returns:
            List of SFT samples
        """
        max_samples = max_samples or self.config.max_sft_samples

        print(f"Loading raw data from {input_path}...")
        raw_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line))

        print(f"Loaded {len(raw_data)} raw samples")

        # Sample if needed
        if len(raw_data) > max_samples:
            raw_data = random.sample(raw_data, max_samples)

        # Construct SFT samples
        print(f"Constructing {len(raw_data)} SFT samples...")
        sft_samples = []
        for item in raw_data:
            sample = self.construct_sft_sample(item)
            if sample:
                sft_samples.append(sample)

        print(f"Created {len(sft_samples)} valid SFT samples")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Saved SFT data to {output_path}")

        return sft_samples


# ============================================================================
# SFT Dataset for Training (无改动)
# ============================================================================

class PersonaMemSFTDataset(Dataset):
    """PyTorch Dataset for SFT training"""

    def __init__(self,
                 data_path: str,
                 tokenizer: Any,
                 max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} SFT samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Format as chat
        messages = sample['messages']
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


# ============================================================================
# SFT Trainer (核心修改：集成 LoRA)
# ============================================================================

class PersonaMemSFTTrainer:
    """
    SFT Trainer for PersonaMem (LoRA 版)

    Uses LoRA to train only adapter parameters, saving massive memory
    """

    def __init__(self, config: PersonaMemSFTConfig = None):
        self.config = config or PersonaMemSFTConfig()

        if not HAS_TORCH or not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("LoRA training requires torch, transformers and peft")

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lora_model = None  # LoRA 包装后的模型

    def load_model(self):
        """Load model + tokenizer + apply LoRA"""
        print(f"Loading model: {self.config.model_name}")

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 【核心修改】配置 4-bit 量化参数，极大降低显存占用
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # 保持计算精度
            bnb_4bit_quant_type="nf4",             # 使用规范浮点量化
            bnb_4bit_use_double_quant=True         # 二次量化节省更多显存
        )

        # 3. 加载基础模型，务必传入 quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map={"": self.config.device},    # 强制指定加载到 cuda:7
            trust_remote_code=True
        )

        # 4. 准备量化训练
        self.model = prepare_model_for_kbit_training(self.model)

        # 5. LoRA 配置
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 6. 获取 LoRA 模型
        self.lora_model = get_peft_model(self.model, lora_config)
        # 再次确保模型完全在目标设备（解决图1的 RuntimeError）
        self.lora_model.to(self.config.device)
        self.lora_model.print_trainable_parameters()

        # 7. 优化器
        self.optimizer = AdamW(self.lora_model.parameters(), lr=self.config.learning_rate)

        print("Model + LoRA loaded successfully")

    def train(self, sft_data_path: str):
        """
        Train LoRA adapter on SFT data

        Args:
            sft_data_path: Path to SFT training data
        """
        if self.lora_model is None:
            self.load_model()

        # Create dataset
        dataset = PersonaMemSFTDataset(
            sft_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Training setup
        total_steps = len(dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Starting LoRA SFT training...")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")

        self.lora_model.train()
        global_step = 0
        accumulated_loss = 0.0

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                # Forward pass (使用 LoRA 模型)
                outputs = self.lora_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update weights (仅更新 LoRA 参数)
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.lora_model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    if global_step % 10 == 0:
                        avg_loss = accumulated_loss / 10
                        print(f"  Step {global_step}: loss = {avg_loss:.4f}")
                        accumulated_loss = 0.0

        # Save LoRA adapter (仅保存适配器，不保存完整模型)
        self.save_model()

    def save_model(self):
        """Save LoRA adapter (轻量化，仅几MB)"""
        output_path = Path(self.config.output_dir) / "lora_adapter"
        output_path.mkdir(parents=True, exist_ok=True)

        self.lora_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        print(f"LoRA adapter saved to {output_path} (only adapter weights, ~MB level)")

    def merge_lora(self, output_path: str = "./output/personamem_sft_merged"):
        """可选：合并 LoRA 适配器到基础模型（推理时使用）"""
        if self.lora_model is None:
            raise ValueError("Train first before merging LoRA")

        # 合并模型
        merged_model = self.lora_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")


# ============================================================================
# Main Entry Points (适配 LoRA 配置)
# ============================================================================

def construct_sft_data(
    input_path: str = "./data/personamem/train_2irrel.jsonl",
    output_path: str = "./data/personamem/sft_train.jsonl",
    max_samples: int = 2000
):
    """Construct SFT training data"""
    config = PersonaMemSFTConfig(
        train_data_path=input_path,
        output_sft_data_path=output_path,
        max_sft_samples=max_samples
    )

    constructor = SFTDataConstructor(config)
    samples = constructor.construct_dataset(input_path, output_path, max_samples)

    # Show example
    if samples:
        print("\n" + "=" * 60)
        print("Example SFT Sample")
        print("=" * 60)
        example = samples[0]
        print(f"\n[Prompt] (truncated):\n{example['prompt'][:500]}...")
        print(f"\n[Response]:\n{example['response']}")

    return samples


def train_sft(
    sft_data_path: str = "./data/personamem/sft_train.jsonl",
    output_dir: str = "./output/personamem_sft_lora",
    model_name: str = "Qwen/Qwen3-4B",
    num_epochs: int = 2
):
    """Train LoRA SFT model"""
    config = PersonaMemSFTConfig(
        output_sft_data_path=sft_data_path,
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=num_epochs
    )

    trainer = PersonaMemSFTTrainer(config)
    trainer.train(sft_data_path)
    # 可选：合并 LoRA 到基础模型
    # trainer.merge_lora()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="PersonaMem SFT (LoRA)")
    parser.add_argument("--mode", type=str, choices=["construct", "train", "both"],
                        default="construct", help="Mode: construct data, train, or both")
    parser.add_argument("--input", type=str, default="./data/personamem/train_2irrel.jsonl",
                        help="Input training data path")
    parser.add_argument("--output", type=str, default="./data/personamem/sft_train.jsonl",
                        help="Output SFT data path")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Maximum SFT samples")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Model name")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default="./output/personamem_sft_lora",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit to save more memory")

    args = parser.parse_args()

    if args.mode in ["construct", "both"]:
        construct_sft_data(args.input, args.output, args.max_samples)

    if args.mode in ["train", "both"]:
        # 传递 4-bit 配置
        config = PersonaMemSFTConfig(
            output_sft_data_path=args.output,
            output_dir=args.output_dir,
            model_name=args.model,
            num_epochs=args.epochs,
            load_in_4bit=args.load_in_4bit
        )
        trainer = PersonaMemSFTTrainer(config)
        trainer.train(args.output)


if __name__ == "__main__":
    main()