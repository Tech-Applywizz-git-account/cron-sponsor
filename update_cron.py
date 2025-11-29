import os
from sqlalchemy import create_engine, text
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import boto3
import json
import re
from datetime import datetime, timedelta

load_dotenv()

# Connect to PostgreSQL
try:
    conn_str = os.environ.get("PSQL_KEY")
    if not conn_str:
        raise ValueError("PSQL_KEY environment variable is missing.")

    # Create SQLAlchemy engine
    engine = create_engine(conn_str)
    print("Successfully connected to PostgreSQL!")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    exit()

# Connect to Supabase
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

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

# System instruction for LLM
SYSTEM_INSTRUCTION = (
    "INSTRUCTION: You are a strict JSON generator. Output ONLY a JSON array of objects as specified. "
    "No explanations, no markdown, no prefixes, no suffix text. If unsure, return []."
)

# Initialize AWS Bedrock client
try:
    bedrock_client = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print(f"Successfully connected to AWS Bedrock! (Region: {AWS_REGION}, Model: {BEDROCK_MODEL_ID})")
except Exception as e:
    print(f"Error connecting to AWS Bedrock: {e}")
    exit()  # Exit if connection to AWS Bedrock fails

def normalize_text(x):
    """Normalize text for processing."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def create_visa_detection_prompt(job_id, title, description):
    """Create a prompt for AWS Bedrock to determine visa sponsorship."""
    # Combine all available text fields
    text_fields = [
        normalize_text(title),
        normalize_text(description)
    ]
    job_text = " ".join([field for field in text_fields if field])
    
    if not job_text.strip():
        return None
    
    # Create job details JSON
    job_details = [{
        "jobId": job_id,
        "title": title if title else "N/A",
        "description": job_text
    }]
    
    prompt = """You are an expert job screening AI specialized in identifying whether a job listing explicitly or implicitly offers valid work visa sponsorship in the United States.

Your task:

- Evaluate each job description in the provided JSON array.
- Determine if the employer is likely to sponsor a U.S. work visa (e.g., H-1B, TN, Skilled Worker, e.t.c.).
- Return a structured JSON array following the exact schema below.
- Do not include explanations, reasoning, or extra text — output valid JSON only.

Input (JSON array of job details):

```json
""" + json.dumps(job_details) + """
Output JSON schema:

[
{"jobId": "string", "sponsorship": "Yes" | "No"}
]

Rules:

- Return "Yes" if the job description explicitly states sponsorship (e.g., "we sponsor H-1B", "visa sponsorship available", "will sponsor work visa").
- Return "No" if the posting explicitly requires existing U.S. work authorization, U.S. citizenship, permanent residency, or states that sponsorship is not available.
- Return "Yes" if the posting is ambiguous but still mentions anything related to international applicants, such as: "may consider international candidates", 
"open to international applicants", "case-by-case visa consideration", "global applicants welcome", In all such cases, treat the job as sponsorship-friendly.
- Return "No" only when the posting does not mention ANYTHING about: sponsorship, work authorization, international candidates, visa considerations, (i.e., complete silence means "No")
- If the job location is clearly outside the United States (e.g., London, Toronto, Bengaluru), return "No" unless the text explicitly says the employer sponsors U.S. work visas.

Constraints:
- Output must be pure JSON. No markdown, no commentary, no text before or after the JSON array."""
    
    return prompt

def call_bedrock_llm(bedrock_client, prompt, model_id=None):
    """Invoke Bedrock model and parse strict JSON array.
    
    Attempts repair on second try. On failure, returns "No".
    """
    if model_id is None:
        model_id = BEDROCK_MODEL_ID
    
    def _invoke(body_text: str):
        """Invoke Bedrock and extract generated text from response."""
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body_text,
            )
            raw_response = response["body"].read().decode()
            
            # Attempt to normalize common Nova response formats
            try:
                wrapper = json.loads(raw_response)
            except Exception as parse_err:
                return raw_response, ""
            
            generated = ""
            if isinstance(wrapper, dict):
                if "results" in wrapper and wrapper["results"]:
                    generated = wrapper["results"][0].get("outputText", "")
                elif "output" in wrapper:
                    out = wrapper["output"]
                    if isinstance(out, dict):
                        msg = out.get("message", {})
                        content = msg.get("content", [])
                        if content and isinstance(content, list):
                            generated = content[0].get("text", "")
                    else:
                        generated = str(out)
                elif "content" in wrapper:
                    content = wrapper["content"]
                    if isinstance(content, list) and content:
                        generated = content[0].get("text", "")
                elif "message" in wrapper:
                    content = wrapper["message"].get("content", [])
                    if content and isinstance(content, list):
                        generated = content[0].get("text", "")
            
            if not generated:
                # As a last resort, search for any nested text fields
                def extract_text_recursive(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if any(tok in k.lower() for tok in ["text", "content", "output"]):
                                res = extract_text_recursive(v)
                                if res:
                                    return res
                        for v in obj.values():
                            res = extract_text_recursive(v)
                            if res:
                                return res
                    if isinstance(obj, list):
                        for it in obj:
                            res = extract_text_recursive(it)
                            if res:
                                return res
                    return ""
                generated = extract_text_recursive(wrapper)
            
            return raw_response, generated or ""
        except Exception as e:
            raise
    
    def _clean(text: str) -> str:
        """Remove markdown code blocks and clean text."""
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9]*", "", t).rstrip("`")
        return t.strip()
    
    def _parse_array(text: str):
        """Parse JSON array from text, with regex fallback."""
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to extract JSON array using regex
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            frag = m.group(0)
            try:
                return json.loads(frag)
            except Exception:
                return None
        return None
    
    # Base request for Nova models (primary format)
    # Note: Nova models don't support "system" role, so prepend system instruction to user message
    full_prompt = SYSTEM_INSTRUCTION + "\n\n" + prompt
    base_request = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": full_prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 4096,
            "temperature": 0.1,
            "topP": 0.9
        }
    }
    
    attempts = 2
    last = ""
    
    for i in range(attempts):
        body = base_request
        if i == 1:
            # Repair prompt on second attempt
            repair = "Your previous response was not valid pure JSON. Return ONLY a JSON array now. No prose. If no data, return [].\n\n"
            repair_prompt = SYSTEM_INSTRUCTION + "\n\n" + repair + prompt
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": repair_prompt}]
                    }
                ],
                "inferenceConfig": base_request["inferenceConfig"]
            }
        
        body_text = json.dumps(body)
        
        try:
            _, out = _invoke(body_text)
            out = _clean(out)
            
            if not out:
                continue
            
            parsed = _parse_array(out)
            if isinstance(parsed, list):
                # Extract sponsorship from first item
                if len(parsed) > 0:
                    sponsorship = parsed[0].get("sponsorship", "No")
                    # Normalize to Yes/No
                    if sponsorship.lower() in ["yes", "y"]:
                        return "Yes"
                    else:
                        return "No"
                else:
                    # Empty array
                    return "No"

            last = out
        except Exception as e:
            if i == attempts - 1:
                return "No"
            continue
    
    # All attempts failed
    return "No"

def detect_visa_sponsorship(row):
    """Detect visa sponsorship using AWS Bedrock LLM."""
    # Extract job information from row
    job_id = str(row.get("job_id", ""))
    title = row.get("title", "")
    description = row.get("description", "")
    
    # Check if we have any text to analyze
    if pd.isna(title) and pd.isna(description):
        return "No"
    
    prompt = create_visa_detection_prompt(job_id, title, description)
    
    if prompt is None:
        return "No"
    
    try:
        result = call_bedrock_llm(bedrock_client, prompt)
        return result
    except Exception as e:
        print(f"Error calling LLM for job {job_id}: {e}")
        return "No"

# Exception handling for fetching max upload date
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
    
    # Convert max_upload_date to datetime object
    if isinstance(max_upload_date, str):
        # Handle ISO format string (with or without timezone)
        if 'T' in max_upload_date:
            max_upload_date_dt = datetime.fromisoformat(max_upload_date.replace('Z', '+00:00'))
        else:
            max_upload_date_dt = datetime.fromisoformat(max_upload_date)
    elif hasattr(max_upload_date, 'isoformat'):
        max_upload_date_dt = max_upload_date
    else:
        max_upload_date_dt = datetime.fromisoformat(str(max_upload_date))
    
    # Calculate end date (max_upload_date + 1 day)
    end_date_dt = max_upload_date_dt + timedelta(days=1)
    
    # Format dates for SQL query
    max_upload_date_str = max_upload_date_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date_dt.strftime('%Y-%m-%d %H:%M:%S')
    
except Exception as e:
    print(f"Error fetching max upload date: {e}")
    exit()  # Exit if fetching max upload date fails

# Fetch a few jobs from the database for testing
# Using the same query structure as update_cron.py, but with ORDER BY and LIMIT
sql_query = f"""
SELECT 
    j.id AS job_id,
    jr.name AS job_role_name,
    j.title,
    j.company,
    j.location,
    j.url,
    j.description,
    j."datePosted" AS date_posted,
    j."yearsExpRequired" AS years_exp_required,
    j."uploadDate" AS upload_date
FROM "karmafy_job" j
LEFT JOIN "karmafy_jobrole" jr
       ON j."roleId"::bigint = jr.id
WHERE j."uploadDate" > '{max_upload_date_str}'
  AND j."uploadDate" <= CURRENT_TIMESTAMP
  AND (jr.name IS NULL OR jr.name NOT LIKE '% for %')
ORDER BY j."uploadDate" DESC;
"""

try:
    print("Fetching data from the psql database...")
    # Use text() to properly handle the SQL query
    df = pd.read_sql(text(sql_query), engine)
    print(f"Fetched {len(df):,} rows from PostgreSQL database.\n")
except Exception as e:
    print(f"Error fetching data from PostgreSQL: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Process jobs with LLM calls
print("=" * 80)
print("PROCESSING JOBS WITH LLM")
print("=" * 80)
print()

results = []

for idx, row in df.iterrows():
    # Use job ID directly from database
    job_id = str(row.get("job_id", ""))
    title = row.get("title", "")
    description = row.get("description", "")
    
    print(f"\n{'='*80}")
    print(f"JOB #{idx + 1} / {len(df)}")
    print(f"{'='*80}")
    print(f"Title: {title}")
    print(f"Company: {row.get('company', 'N/A')}")
    print(f"Location: {row.get('location', 'N/A')}")
    print(f"Job ID: {job_id}")
    print(f"\nCalling LLM for visa sponsorship detection...")
    
    # Call LLM to detect visa sponsorship
    sponsorship_result = detect_visa_sponsorship(row)
    
    print(f"Result: {sponsorship_result}")
    
    # Store result
    results.append({
        "job_id": job_id,
        "title": title,
        "company": row.get('company', 'N/A'),
        "location": row.get('location', 'N/A'),
        "sponsorship": sponsorship_result
    })
    
    print(f"{'-'*80}")

print("\n" + "=" * 80)
print("LLM PROCESSING COMPLETE")
print("=" * 80)
print(f"\nTotal jobs processed: {len(df)}")
print(f"\nResults Summary:")
print("-" * 80)
sponsored_count = sum(1 for r in results if r["sponsorship"] == "Yes")
print(f"Sponsored jobs: {sponsored_count}")
print(f"Non-sponsored jobs: {len(results) - sponsored_count}")
print(f"\nDetailed Results (Sponsored Jobs Only):")
print("-" * 80)
for result in results:
    if result["sponsorship"] == "Yes":
        print(f"Job ID: {result['job_id']} | {result['title']} | Sponsorship: {result['sponsorship']}")

# Prepare sponsored jobs for Supabase insertion
sponsored_results = [r for r in results if r["sponsorship"] == "Yes"]

if len(sponsored_results) > 0:
    print("\n" + "=" * 80)
    print("PREPARING SPONSORED JOBS FOR SUPABASE INSERTION")
    print("=" * 80)
    
    # Create a dataframe from sponsored results and merge with original df data
    sponsored_jobs_data = []
    for result in sponsored_results:
        # Find the corresponding row in the original dataframe
        job_id = result["job_id"]
        original_row = df[df["job_id"].astype(str) == str(job_id)].iloc[0] if len(df[df["job_id"].astype(str) == str(job_id)]) > 0 else None
        
        if original_row is not None:
            # Prepare data for Supabase (excluding job_id as it's auto-incremental)
            # Match the exact column structure from the example
            job_data = {
                "job_role_name": original_row.get("job_role_name") if pd.notna(original_row.get("job_role_name")) else None,
                "title": original_row.get("title") if pd.notna(original_row.get("title")) else None,
                "company": original_row.get("company") if pd.notna(original_row.get("company")) else None,
                "location": original_row.get("location") if pd.notna(original_row.get("location")) else None,
                "url": original_row.get("url") if pd.notna(original_row.get("url")) else None,
                "description": original_row.get("description") if pd.notna(original_row.get("description")) else None,
                "date_posted": original_row.get("date_posted").isoformat() if pd.notna(original_row.get("date_posted")) else None,
                "years_exp_required": original_row.get("years_exp_required") if pd.notna(original_row.get("years_exp_required")) else None,
                "upload_date": original_row.get("upload_date").isoformat() if pd.notna(original_row.get("upload_date")) else None,
                "sponsored_job": "Yes",
                "country": "United States of America"
            }
            sponsored_jobs_data.append(job_data)
    
    if len(sponsored_jobs_data) > 0:
        # Replace NaN/None values with None (Supabase-friendly)
        for record in sponsored_jobs_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        # Insert data in batches if needed (Supabase has limits)
        table_name = "job_jobrole_sponsored"
        batch_size = 1000
        total_inserted = 0
        
        print(f"\nInserting {len(sponsored_jobs_data)} sponsored jobs into Supabase...")
        
        try:
            for i in range(0, len(sponsored_jobs_data), batch_size):
                batch = sponsored_jobs_data[i:i + batch_size]
                if batch:  # Ensure batch is not empty
                    response = supabase.table(table_name).insert(batch).execute()
                    total_inserted += len(response.data) if response.data else 0
                    print(f"Inserted batch {i // batch_size + 1}: {len(batch)} rows")
            
            print(f"\n{'='*80}")
            print(f"SUPABASE INSERTION COMPLETE")
            print(f"{'='*80}")
            print(f"Total rows inserted: {total_inserted}")
        except Exception as e:
            print(f"\nError inserting data to Supabase: {e}")
            print(f"Attempted to insert {len(sponsored_jobs_data)} rows")
            print(f"Columns in data: {list(sponsored_jobs_data[0].keys()) if sponsored_jobs_data else 'N/A'}")
            import traceback
            traceback.print_exc()
    else:
        print("No sponsored jobs data prepared for insertion.")
else:
    print("\nNo sponsored jobs to insert into Supabase.")

