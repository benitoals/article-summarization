# debug_dataset_v2.py

from datasets import load_dataset
from transformers import AutoTokenizer

def debug_dataset(dataset_repo_id="benitoals/my-txt-dataset", num_samples=5):
    dataset = load_dataset(dataset_repo_id)
    print("Loaded dataset splits:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} examples")

    print("\nDataset features for train split:")
    print(dataset["train"].features)

    print("\n--- First few examples from the train split ---")
    for i in range(num_samples):
        ex = dataset["train"][i]
        print(f"\nExample {i+1}:")
        # ex should be a dict with keys "filename", "body", "summary"
        print("Filename:", ex.get("filename", "N/A"))
        body = ex.get("body", "").strip()
        summary = ex.get("summary", "").strip()
        print("Body (first 500 chars):")
        print(body[:500] if body else "Empty body")
        print("\nSummary (first 500 chars):")
        print(summary[:500] if summary else "Empty summary")
        print("-" * 80)

    # Count rows with header values if any
    header_row = {"filename": "filename", "body": "body", "summary": "summary"}
    header_count = sum(
        1 for i in range(len(dataset["train"]))
        if dataset["train"][i] == header_row
    )
    print(f"\nHeader row count in train: {header_count}")

    # Tokenization debugging
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=True)
    print("\n--- Tokenization Debugging ---")
    for i in range(num_samples):
        ex = dataset["train"][i]
        body = ex.get("body", "")
        summary = ex.get("summary", "")
        body_input = "summarize: " + body
        tokens_body = tokenizer.tokenize(body_input)
        tokens_summary = tokenizer.tokenize(summary)
        print(f"\nExample {i+1} tokenization:")
        print("Body tokens (first 30):", tokens_body[:30])
        print("Body token count:", len(tokens_body))
        print("Summary tokens (first 30):", tokens_summary[:30])
        print("Summary token count:", len(tokens_summary))
        print("-" * 80)


from datasets import load_dataset
from transformers import AutoTokenizer

def search_extremes_in_dataset(dataset_repo_id="benitoals/my-txt-dataset", body_key="body", summary_key="summary", num_extremes=3):
    """
    Loads the dataset from Hugging Face Hub, computes token counts for the body and summary using the specified tokenizer,
    and prints the top 'num_extremes' examples with highest and lowest token counts.
    """
    # Load dataset
    dataset = load_dataset(dataset_repo_id)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=True)
    
    token_info = []
    for i, ex in enumerate(dataset["train"]):
        body = ex.get(body_key, "")
        summary = ex.get(summary_key, "")
        # Prepend "summarize: " to body if needed (consistent with your preprocessing)
        tokens_body = tokenizer.tokenize("summarize: " + body)
        tokens_summary = tokenizer.tokenize(summary)
        token_info.append({
            "index": i,
            "filename": ex.get("filename", "N/A"),
            "body_token_count": len(tokens_body),
            "summary_token_count": len(tokens_summary)
        })
    
    # Sort to find extremes
    sorted_by_body_desc = sorted(token_info, key=lambda x: x["body_token_count"], reverse=True)
    sorted_by_summary_desc = sorted(token_info, key=lambda x: x["summary_token_count"], reverse=True)
    
    sorted_by_body_asc = sorted(token_info, key=lambda x: x["body_token_count"])
    sorted_by_summary_asc = sorted(token_info, key=lambda x: x["summary_token_count"])
    
    print("\nTop {} examples with highest BODY token count:".format(num_extremes))
    for info in sorted_by_body_desc[:num_extremes]:
        print(f"Index {info['index']} - Filename: {info['filename']} - Body token count: {info['body_token_count']}")
    
    print("\nTop {} examples with highest SUMMARY token count:".format(num_extremes))
    for info in sorted_by_summary_desc[:num_extremes]:
        print(f"Index {info['index']} - Filename: {info['filename']} - Summary token count: {info['summary_token_count']}")
    
    print("\nTop {} examples with lowest BODY token count:".format(num_extremes))
    for info in sorted_by_body_asc[:num_extremes]:
        print(f"Index {info['index']} - Filename: {info['filename']} - Body token count: {info['body_token_count']}")
    
    print("\nTop {} examples with lowest SUMMARY token count:".format(num_extremes))
    for info in sorted_by_summary_asc[:num_extremes]:
        print(f"Index {info['index']} - Filename: {info['filename']} - Summary token count: {info['summary_token_count']}")

if __name__ == "__main__":
    # You can call this function after your existing debug output:
    # debug_dataset()
    search_extremes_in_dataset()
