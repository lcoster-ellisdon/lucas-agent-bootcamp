import os
import csv
from dotenv import load_dotenv
from pageindex import PageIndexClient
import pageindex.utils as utils
load_dotenv()

# ==============================
# Configuration
# ==============================

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
DATASET_FOLDER = "../../datasets"   # folder where your files are stored
OUTPUT_CSV = "uploaded_documents.csv"

# ==============================
# Initialize PageIndex Client
# ==============================

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# ==============================
# Upload Files & Store Mapping
# ==============================

def upload_documents():
    results = []

    # Loop through all files in dataset folder
    for file_name in os.listdir(DATASET_FOLDER):

        file_path = os.path.join(DATASET_FOLDER, file_name)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        try:
            print(f"\nUploading: {file_name}")

            response = pi_client.submit_document(file_path)
            doc_id = response["doc_id"]

            print(f"Document Submitted: {doc_id}")

            # Store mapping
            results.append({
                "file_name": file_name,
                "doc_id": doc_id
            })

        except Exception as e:
            print(f"Error uploading {file_name}: {str(e)}")

    return results


# ==============================
# Save Results to CSV
# ==============================

def save_to_csv(results):
    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_name", "doc_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file does not exist
        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nSaved file_name ↔ doc_id mapping to {OUTPUT_CSV}")


# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    uploaded_results = upload_documents()
    save_to_csv(uploaded_results)
