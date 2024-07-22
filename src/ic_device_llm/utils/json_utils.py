import json
from typing import List, Dict


def load_jsonl_files(jsonl_file_path: str) -> List[Dict]:
    """
    Load a jsonl file into a list of dictionaries.
    Args:
        jsonl_file_path: The path to the jsonl file.
    Returns:
        A list of dictionaries.
    """
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
