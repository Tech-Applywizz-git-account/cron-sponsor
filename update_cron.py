import os
from sqlalchemy import create_engine, text
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import boto3
import json
import re
from datetime import datetime, timedelta
import concurrent.futures  # Added for concurrency

load_dotenv()

# Connect to PostgreSQL
try:
    conn_str = os.environ.get("PSQL_KEY")
    if not conn_str:
        raise ValueError("PSQL_KEY environment variable is missing.")

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
    exit()

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
    exit()

# [Keep all helper functions unchanged: normalize_text, create_visa_detection_prompt, call_bedrock_llm, detect_visa_sponsorship]
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
    text_fields = [
        normalize_text(title),
        normalize_text(description)
    ]
    job_text = " ".join([field for field in text_fields if field])
    
    if not job_text.strip():
        return None
    
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
                if len(parsed) > 0:
                    sponsorship = parsed[0].get("sponsorship", "No")
                    if sponsorship.lower() in ["yes", "y"]:
                        return "Yes"
                    else:
                        return "No"
                else:
                    return "No"

            last = out
        except Exception as e:
            if i == attempts - 1:
                return "No"
            continue
    
    return "No"

def detect_visa_sponsorship(row):
    """Detect visa sponsorship using AWS Bedrock LLM."""
    job_id = str(row.get("job_id", ""))
    title = row.get("title", "")
    description = row.get("description", "")
    
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

# Fixed date
target_date = datetime(2026, 3, 28)

start_date = target_date.replace(hour=0, minute=0, second=0)
end_date = target_date.replace(hour=23, minute=59, second=59)

start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

print(f"Processing jobs from {start_date_str} to {end_date_str}")

# Fetch jobs
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
    j."uploadDate" AS upload_date,
    j.salary,
    j."applyType" AS apply_type
FROM "karmafy_job" j
LEFT JOIN "karmafy_jobrole" jr
       ON j."roleId"::bigint = jr.id
WHERE j."uploadDate" >= '{start_date_str}'
  AND j."uploadDate" <= '{end_date_str}'
  AND (jr.name IS NULL OR jr.name NOT LIKE '% for %')
  AND j.is_staffing = 'No'
ORDER BY j."uploadDate" DESC;
"""

try:
    print("Fetching data from the psql database...")
    df = pd.read_sql(text(sql_query), engine)
    print(f"Fetched {len(df):,} rows from PostgreSQL database.\n")
except Exception as e:
    print(f"Error fetching data from PostgreSQL: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Process jobs concurrently
print("=" * 80)
print("PROCESSING JOBS WITH LLM (concurrent)")
print("=" * 80)
print()

results = []
sponsored_job_ids = []

def process_job(row):
    """Process a single job: call LLM and return result with original row."""
    job_id = str(row.get("job_id", ""))
    sponsorship_result = detect_visa_sponsorship(row)
    return {
        "job_id": job_id,
        "title": row.get("title", ""),
        "company": row.get("company", "N/A"),
        "location": row.get("location", "N/A"),
        "sponsorship": sponsorship_result,
        "original_row": row  # keep for later insertion
    }

# Use ThreadPoolExecutor to parallelize LLM calls
MAX_WORKERS = 10  # Adjust based on your Bedrock limits
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_idx = {executor.submit(process_job, row): idx for idx, row in df.iterrows()}
    completed = 0
    total = len(future_to_idx)
    for future in concurrent.futures.as_completed(future_to_idx):
        idx = future_to_idx[future]
        completed += 1
        try:
            result = future.result()
            results.append(result)
            if result["sponsorship"] == "Yes":
                sponsored_job_ids.append(int(result["job_id"]))
            # Print progress (optional, may interleave)
            print(f"[{completed}/{total}] {result['job_id']}: {result['title'][:60]} -> {result['sponsorship']}")
        except Exception as e:
            print(f"Error processing job index {idx}: {e}")
            # Add a dummy result to keep count consistent
            results.append({
                "job_id": str(df.at[idx, "job_id"]),
                "sponsorship": "No",
                "original_row": df.iloc[idx]
            })

print("\n" + "=" * 80)
print("LLM PROCESSING COMPLETE")
print("=" * 80)

# Bulk update PostgreSQL is_sponsored
if sponsored_job_ids:
    try:
        update_query = text("""
            UPDATE karmafy_job 
            SET is_sponsored = 'Yes' 
            WHERE id = ANY(:job_ids)
        """)
        with engine.connect() as conn:
            conn.execute(update_query, {"job_ids": sponsored_job_ids})
            conn.commit()
        print(f"✓ Updated is_sponsored = 'Yes' for {len(sponsored_job_ids)} jobs")
    except Exception as e:
        print(f"✗ Error updating is_sponsored in PostgreSQL: {e}")
else:
    print("No jobs to update as sponsored.")

# Summary
sponsored_count = sum(1 for r in results if r["sponsorship"] == "Yes")
print(f"\nTotal jobs processed: {len(results)}")
print(f"Sponsored jobs: {sponsored_count}")
print(f"Non-sponsored jobs: {len(results) - sponsored_count}")
print("\nDetailed Results (Sponsored Jobs Only):")
for result in results:
    if result["sponsorship"] == "Yes":
        print(f"Job ID: {result['job_id']} | {result['title']} | Sponsorship: {result['sponsorship']}")

# Prepare sponsored jobs for Supabase
sponsored_results = [r for r in results if r["sponsorship"] == "Yes"]

if sponsored_results:
    print("\n" + "=" * 80)
    print("PREPARING SPONSORED JOBS FOR SUPABASE INSERTION")
    print("=" * 80)
    
    sponsored_jobs_data = []
    for result in sponsored_results:
        original_row = result["original_row"]
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
            "country": "United States of America",
            "jobId": result["job_id"] or None,
            "salary": original_row.get("salary") if pd.notna(original_row.get("salary")) else None,
            "apply_type": original_row.get("apply_type") if pd.notna(original_row.get("apply_type")) else None
        }
        # Replace NaN/None values with None
        for key, value in job_data.items():
            if pd.isna(value):
                job_data[key] = None
        sponsored_jobs_data.append(job_data)
    
    if sponsored_jobs_data:
        table_name = "job_jobrole_sponsored"
        batch_size = 1000
        total_inserted = 0
        
        print(f"Inserting {len(sponsored_jobs_data)} sponsored jobs into Supabase...")
        try:
            for i in range(0, len(sponsored_jobs_data), batch_size):
                batch = sponsored_jobs_data[i:i+batch_size]
                if batch:
                    response = supabase.table(table_name).insert(batch).execute()
                    total_inserted += len(response.data) if response.data else 0
                    print(f"Inserted batch {i // batch_size + 1}: {len(batch)} rows")
            print(f"\nSUPABASE INSERTION COMPLETE: {total_inserted} rows inserted")
        except Exception as e:
            print(f"Error inserting data to Supabase: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No sponsored jobs data prepared for insertion.")
else:
    print("No sponsored jobs to insert into Supabase.")

