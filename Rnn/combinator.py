from pathlib import Path
from datasort import *

# === Combine text from multiple files ===
def load_multiple_texts(file_paths):
    full_text = ""
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        text = extract_text(file_path)
        full_text += text + "\n"  # Adding newlines to separate documents
    return full_text

#all files in the directory
def get_all_files_in_directory(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"The path {directory_path} is not a valid directory.")
    return [str(file) for file in directory.glob("*") if file.is_file()]

#holder for the directory path
directory_path = "/Users/alistairchambers/Machine Learning/stories"
# Get all files in the directory
file_paths = get_all_files_in_directory(directory_path)


# Example usage

combined_text = load_multiple_texts(file_paths)

# Optional: Preprocessing (lowercase, remove special characters)
combined_text = combined_text.lower()

# Now you can proceed with the rest of the model training process

# Save the combined text to a file
output_file_path = "combined_text.txt"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(combined_text)
print(f"Combined text saved to {output_file_path}")
# Save the combined text to a file

