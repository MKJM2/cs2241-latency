import sys
import csv
import os

# --- Configuration ---
INPUT_FILENAME = "malicious_phish.csv"
OUTPUT_FILENAME = "processed_urls.csv"
URL_COLUMN_NAME = "url"  # Expected name in input header
TYPE_COLUMN_NAME = "type"  # Expected name in input header
BENIGN_TYPE_STRING = "benign"  # The string indicating a benign URL
OUTPUT_KEY_COLUMN = "key"
OUTPUT_LABEL_COLUMN = "label"
MALICIOUS_LABEL_VALUE = 1
BENIGN_LABEL_VALUE = 0
# --- End Configuration ---


def convert_csv(input_path, output_path):
    """
    Reads the input CSV, converts URL types to binary labels (0 or 1),
    and writes the result to the output CSV.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV file will be saved.
    """
    print(f"Processing '{input_path}'...")
    processed_count = 0
    skipped_count = 0

    try:
        with (
            open(input_path, mode="r", newline="", encoding="utf-8") as infile,
            open(output_path, mode="w", newline="", encoding="utf-8") as outfile,
        ):
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # --- Header Handling ---
            try:
                header = next(reader)
                print(f"Input Header: {header}")
                # Find column indices (more robust than assuming fixed positions)
                try:
                    url_index = header.index(URL_COLUMN_NAME)
                    type_index = header.index(TYPE_COLUMN_NAME)
                except ValueError as e:
                    print(
                        f"Error: Missing expected column in header: {e}. "
                        f"Expected '{URL_COLUMN_NAME}' and '{TYPE_COLUMN_NAME}'."
                    )
                    return  # Stop processing if header is wrong

            except StopIteration:
                print(f"Error: Input file '{input_path}' is empty.")
                return  # Stop if file is empty

            # Write the new header
            writer.writerow([OUTPUT_KEY_COLUMN, OUTPUT_LABEL_COLUMN])

            # --- Row Processing ---
            for i, row in enumerate(
                reader, start=1
            ):  # Start counting rows from 1 after header
                if len(row) > max(
                    url_index, type_index
                ):  # Check if row has enough columns
                    url = row[url_index]
                    original_type = row[type_index].strip().lower()  # Normalize

                    # Determine the label
                    label = (
                        BENIGN_LABEL_VALUE
                        if original_type == BENIGN_TYPE_STRING
                        else MALICIOUS_LABEL_VALUE
                    )

                    # Write the output row
                    writer.writerow([url, label])
                    processed_count += 1
                else:
                    print(
                        f"Warning: Skipping malformed row #{i + 1} (not enough columns): {row}"
                    )
                    skipped_count += 1

        print(f"\nSuccessfully processed {processed_count} rows.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} malformed rows.")
        print(f"Output saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    # Check if input file exists before starting
    if not os.path.exists(INPUT_FILENAME):
        print(
            f"Error: The input file '{INPUT_FILENAME}' was not found in the current directory."
        )
        sys.exit(1)

    convert_csv(INPUT_FILENAME, OUTPUT_FILENAME)
