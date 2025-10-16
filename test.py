import pandas as pd
from datetime import datetime
import json

# Sample data with a Timestamp (datetime) field
data = {
    "job_role_name": ["Software Engineer"],
    "title": ["Senior Developer"],
    "company": ["Tech Corp"],
    "location": ["New York"],
    "url": ["https://example.com"],
    "description": ["A senior role in software engineering."],
    "raw_text": ["Some raw text here."],
    "date_posted": [datetime(2025, 10, 15, 5, 55, 4, 657699)],  # Timestamp object
    "upload_date": [datetime(2025, 10, 15, 5, 55, 4, 657699)],  # Timestamp object
    "sponsored_job": [True],
}

# Create a DataFrame
df = pd.DataFrame(data)


# Function to check if Timestamp can be serialized into JSON
def check_serializability(df):
    try:
        # Attempt to convert Timestamp columns to ISO format strings
        df["upload_date"] = df["upload_date"].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        )
        df["date_posted"] = df["date_posted"].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        )

        # Try serializing the DataFrame to JSON format
        data_to_insert = df.to_dict(orient="records")
        json_data = json.dumps(
            data_to_insert
        )  # This will raise an error if it's not serializable

        print("Data successfully serialized into JSON!")
        return json_data

    except TypeError as e:
        print(f"Error serializing data: {e}")
        return None


# Run the check until data becomes serializable
serializable = False
attempts = 0
while not serializable:
    attempts += 1
    print(f"Attempt {attempts}...")

    # Try to serialize the data
    result = check_serializability(df)

    if result:
        serializable = True
    else:
        # If serialization failed, convert to Unix timestamp and try again
        print("Converting timestamps to Unix timestamps and retrying...")
        df["upload_date"] = df["upload_date"].apply(
            lambda x: int(x.timestamp()) if pd.notna(x) else None
        )
        df["date_posted"] = df["date_posted"].apply(
            lambda x: int(x.timestamp()) if pd.notna(x) else None
        )

print("Final JSON data (serialized):")
print(result)
