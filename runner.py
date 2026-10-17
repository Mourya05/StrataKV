"""Run the StrataKV token bridge and decode its output with TinyLlama's tokenizer."""

from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MESSAGES = [
    {"role": "system", "content": "You are a highly capable backend engineering assistant."},
    {"role": "user", "content": "What is a paged key-value cache in a transformer?"}
]
MAX_NEW_TOKENS = 500


def find_cli(repository_root: Path) -> Path:
    candidates = (
        repository_root / "build" / "Release" / "stratakv_cli.exe",
        repository_root / "build" / "Release" / "stratakv_cli",
        repository_root / "build" / "stratakv_cli.exe",
        repository_root / "build" / "stratakv_cli",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"stratakv_cli was not found; checked: {expected}")


def main() -> int:
    repository_root = Path(__file__).resolve().parent
    model_path = repository_root / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")

    cli_path = find_cli(repository_root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_prompt_data = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, return_dict=True)
    prompt_ids = raw_prompt_data["input_ids"]

    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    command = [
        str(cli_path),
        str(model_path),
        str(MAX_NEW_TOKENS),
        *(str(token_id) for token_id in prompt_ids),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        return error.returncode or 1

    raw_ids = completed.stdout.split()
    if not raw_ids:
        raise RuntimeError("stratakv_cli returned no token IDs")
    try:
        output_ids = [int(token_id) for token_id in raw_ids]
    except ValueError as error:
        raise RuntimeError(
            "stratakv_cli returned non-integer stdout: " + completed.stdout
        ) from error

    print(tokenizer.decode(output_ids, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - user-facing runner boundary.
        print(f"runner.py: {error}", file=sys.stderr)
        raise SystemExit(1)
