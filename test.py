import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

try:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Supabase URL or KEY is missing.")

    supabase = create_client(url, key)
    print("Successfully connected to Supabase!")
except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    exit()  # Exit if connection to Supabase fails

try:
    response = (
        supabase.table("job_jobrole_sponsored")
        .select("upload_date")
        .not_.is_("upload_date", None)
        .order("upload_date", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise ValueError("No data returned for max upload date.")

    max_upload_date = response.data[0]["upload_date"]
    print(f"Max upload date: {max_upload_date}")
except Exception as e:
    print(f"Error fetching max upload date: {e}")
    exit()  # Exit if fetching max upload date fails
