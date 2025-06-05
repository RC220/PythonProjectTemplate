import argparse
import logging
import os
import glob
import re
from collections import Counter
import pandas as pd
from tqdm.contrib.concurrent import process_map
from pathlib import Path

logger = logging.getLogger(__name__)

def tokenize(text, pattern=r"\s"):
    """Returns a list of strings, the text split into tokens based on the regex pattern to identify boundaries."""
    tokenized = re.split(pattern, text)
    return tokenized

def process_file(path_and_flag):
    file_path, case_insensitive = path_and_flag
    try:
        text = Path(file_path).read_text(encoding="utf-8")
        tokens = tokenize(text)
        # Filter out empty tokens
        non_empty_tokens = [w for w in tokens if w != ""]
        
        # Handle case insensitivity at counting stage
        if case_insensitive:
            # Convert all tokens to lowercase before counting
            non_empty_tokens = [w.lower() for w in non_empty_tokens]
            logger.debug(f"Case insensitive tokens for {file_path}: {non_empty_tokens}")
        else:
            logger.debug(f"Case sensitive tokens for {file_path}: {non_empty_tokens}")
            
        counter = Counter(non_empty_tokens)
        logger.debug(f"Counter for {file_path}: {dict(counter)}")
        return counter
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return Counter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document_dir")
    parser.add_argument("csv_out")
    parser.add_argument(
        "--case-insensitive", "--case_insensitive",
        action="store_true",
        default=False,
        dest="case_insensitive"
    )
    parser.add_argument(
        "--workers", type=int, default=4
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    pattern = os.path.join(args.document_dir, "*.txt")
    files = glob.glob(pattern)
    pairs = [(fp, args.case_insensitive) for fp in files]

    results = process_map(
        process_file,
        pairs,
        max_workers=args.workers,
        chunksize=1
    )

    # Combine all counters
    final_counter = Counter()
    for counter in results:
        final_counter.update(counter)
        if args.debug:
            logger.debug(f"Current final counter: {dict(final_counter)}")

    # Create and save DataFrame
    df = pd.DataFrame.from_records(
        list(final_counter.items()), 
        columns=["token", "count"]
    ).sort_values(by="token")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    df.to_csv(args.csv_out, index=False)

if __name__ == "__main__":
    main()
