from google.cloud import logging as cloud_logging

PROJECT_ID = "aisprint-491218"
try:
    client = cloud_logging.Client(project=PROJECT_ID)
    print(f"Testing logging for project {PROJECT_ID}")
    entries = list(client.list_entries(page_size=1))
    print(f"Success! Found {len(entries)} entries.")
except Exception as e:
    print(f"Error: {e}")
