"""Benchmark StrataKV generation throughput."""

import time
import subprocess
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MESSAGES = [
    {"role": "system", "content": "You are a highly capable backend engineering assistant."},
    {"role": "user", "content": "Explain the architecture of a GPU. Be extremely detailed."}
]
MAX_NEW_TOKENS = 500

def main():
    repository_root = Path(__file__).resolve().parent
    model_path = repository_root / "model.safetensors"
    cli_path = repository_root / "build" / "Release" / "stratakv_cli.exe"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_prompt_data = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, return_dict=True)
    prompt_ids = raw_prompt_data["input_ids"]

    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    command = [
        str(cli_path),
        str(model_path),
        str(MAX_NEW_TOKENS),
        "--temp", "0.8",
        "--top-k", "40",
        "--top-p", "0.95",
        *(str(tid) for tid in prompt_ids)
    ]

    print(f"Benckmarking StrataKV Engine (Target: {MAX_NEW_TOKENS} tokens)...")
    
    start_time = time.perf_counter()
    
    completed = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time

    raw_ids = completed.stdout.split()
    output_tokens = len(raw_ids)
    
    if output_tokens == 0:
        print("Error: Engine returned no tokens.")
        return

    # Subtract prompt tokens from total output to get generated tokens
    generated_count = output_tokens - len(prompt_ids)
    tps = generated_count / total_time

    print("\n--- Benchmark Results ---")
    print(f"Total Wall Time:  {total_time:.2f} seconds")
    print(f"Tokens Generated: {generated_count}")
    print(f"Throughput:       {tps:.2f} tokens/sec")
    print("-------------------------\n")
    
    print("Output Preview:")
    print(tokenizer.decode([int(t) for t in raw_ids[-100:]], skip_special_tokens=True))

if __name__ == "__main__":
    main()