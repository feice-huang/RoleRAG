import json
import sys
import os

def jsonl_to_json(file_path):
    """
    Converts a JSONL file to a JSON array and overwrites the original file.
    
    :param file_path: Path to the JSONL file
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        # Read the JSONL file
        with open(file_path, 'r', encoding='utf-8') as jsonl_file:
            json_array = [json.loads(line) for line in jsonl_file if line.strip()]

        # Write the JSON array back to the same file
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_array, json_file, ensure_ascii=False, indent=4)

        print(f"Successfully converted '{file_path}' to JSON array format.")
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

if __name__ == "__main__":
    jsonl_to_json("/data/hfc/RoleRAG/data/刘星_sft_shuffle.jsonl")