import os
import re
from datasets import Dataset, DatasetDict

# ---------------------------
# TXT Extraction and Cleaning
# ---------------------------

def read_text_file(txt_path):
    """Reads text from a TXT file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""

def naive_find_abstract_and_body(raw_text):
    """
    Extracts the abstract and body from a paper's text.
    
    The body is defined as everything after the abstract (i.e., it excludes the title,
    author information, and abstract). Additionally, any references section (e.g. "References",
    "Bibliography", "Works Cited") is removed from the body.
    """
    lines = raw_text.splitlines()
    # Remove empty lines and trim whitespace
    lines = [line.strip() for line in lines if line.strip()]

    abstract_lines = []
    body_lines = []
    found_abstract = False
    in_abstract = False

    # Regex for abstract header: matches "Abstract" (with variations) at the beginning.
    abstract_pat = re.compile(r"^\s*Abstract[\s:—-]*", re.IGNORECASE)
    # Regex for markers indicating the end of the abstract block.
    next_section_pat = re.compile(
        r"^\s*(?:" +
            r"(?:Index\s+T\s*erms[\s:—-]*)|" +
            r"(?:Keywords\s*-\s*)|" +
            r"(?:Keywords[\s:—-]*)|" +
            r"(?:[0-9]+\s*\.?\s*[Ii]\s*[Nn]\s*[Tt]\s*[Rr]\s*[Oo]\s*[Dd]\s*[Uu]\s*[Cc]\s*[Tt]\s*[Ii]\s*[Oo]\s*[Nn]\b)|" +
            r"(?:I\.\s*Introduction\b)|" +
            r"(?:I[\s\.]+(?:I[\s\.]+)?[Ii]\s*[Nn]\s*[Tt]\s*[Rr]\s*[Oo]\s*[Dd]\s*[Uu]\s*[Cc]\s*[Tt]\s*[Ii]\s*[Oo]\s*[Nn]\b)|" +
            r"(?:I\.\s*I\s+NTRODUCTION\b)" +
        r")",
        re.IGNORECASE
    )

    for line in lines:
        if not found_abstract:
            # Skip lines until we find the abstract header.
            if abstract_pat.match(line):
                found_abstract = True
                in_abstract = True
                # Remove the header (e.g. "Abstract —") from this line.
                header_removed = abstract_pat.sub('', line).strip()
                if header_removed:
                    abstract_lines.append(header_removed)
            # Else: ignore pre-abstract lines (e.g. title, authors)
        else:
            if in_abstract:
                # If we detect a marker (e.g. "I. Introduction"), then the abstract ends.
                if next_section_pat.match(line):
                    in_abstract = False
                    # Include the marker (which is part of the introduction) in the body.
                    body_lines.append(line)
                else:
                    abstract_lines.append(line)
            else:
                # After the abstract, all lines belong to the body.
                body_lines.append(line)

    abstract_txt = " ".join(abstract_lines).strip()
    body_txt = " ".join(body_lines).strip()

    # Remove any references section from the body.
    references_pat = re.compile(r'\b(?:References|Bibliography|Works Cited)\b', re.IGNORECASE)
    ref_match = references_pat.search(body_txt)
    if ref_match:
        body_txt = body_txt[:ref_match.start()].strip()

    return abstract_txt, body_txt
def clean_text(text):
    """
    Cleans text by:
      - Removing hyphenated line breaks.
      - Normalizing newlines and extra spaces.
      - Removing non-ASCII characters (optional).
    """
    if not text:
        return ""
    text = text.replace("-\n", "")
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# ---------------------------
# Build Dataset from TXT Files
# ---------------------------

def build_examples_from_txts(txt_folder):
    """
    Iterates over all TXT files in txt_folder, extracts and cleans the abstract and body.
    Skips files without an abstract.
    """
    txt_files = [f for f in os.listdir(txt_folder) if f.lower().endswith(".txt")]
    data = []
    for txt_file in txt_files:
        path = os.path.join(txt_folder, txt_file)
        print(f"Processing file: {txt_file}")
        raw_text = read_text_file(path)
        if not raw_text:
            print(f"  [DEBUG] No text read from {txt_file}. Skipping.")
            continue
        abstract, body = naive_find_abstract_and_body(raw_text)
        abstract = clean_text(abstract)
        body = clean_text(body)
        if not abstract:
            print(f"  [DEBUG] Skipping {txt_file} as no abstract was found.\n")
            continue
        # Debug output: show first 150 characters of abstract and body
        print(f"  [DEBUG] Abstract (first 150 chars): {abstract[:150]}...")
        print(f"  [DEBUG] Body (first 150 chars): {body[:150]}...\n")
        data.append({
            "filename": txt_file,
            "summary": abstract,
            "body": body
        })
    return data

def split_train_val_test(examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Shuffles and splits the examples into train, validation, and test splits.
    """
    import random
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]
    dset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })
    return dset

def maybe_build_and_push_local_dataset(txt_folder, dataset_repo_id):
    """
    Builds a Hugging Face DatasetDict from TXT files in txt_folder.
    Saves the dataset to "local_txt_dataset" and pushes it to the Hugging Face Hub.
    """
    examples = build_examples_from_txts(txt_folder)
    if len(examples) == 0:
        print("No valid TXT files found or no data extracted. Exiting.")
        return
    dataset = split_train_val_test(examples)
    print("Created dataset splits:", {k: len(dataset[k]) for k in dataset})
    dataset.save_to_disk("local_txt_dataset")
    dataset.push_to_hub(dataset_repo_id)
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_repo_id}")

# ---------------------------
# MAIN
# ---------------------------

def main():
    txt_folder = "sources"              # Folder containing the TXT files
    dataset_repo_id = "benitoals/new-txt-dataset"  # Replace with your HF Hub repo ID
    maybe_build_and_push_local_dataset(txt_folder, dataset_repo_id)

if __name__ == "__main__":
    main()