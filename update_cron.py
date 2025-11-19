import os
from sqlalchemy import create_engine
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import boto3
import json
import re

load_dotenv()

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
except Exception as e:
    print(f"Error fetching max upload date: {e}")
    exit()  # Exit if fetching max upload date fails


sql_query = f"""
SELECT 
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
WHERE j."uploadDate" > '{max_upload_date}'
  AND j."uploadDate" <= CURRENT_TIMESTAMP;
"""


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
    
    prompt = """You are an expert job screening AI specialized in identifying whether a job listing explicitly or implicitly offers valid work visa sponsorship.

Your task:

- Evaluate each job description in the provided JSON array.

- Determine if the employer is likely to sponsor a work visa (e.g., H-1B, TN, Skilled Worker, etc.).

- Return a structured JSON array following the exact schema below.

- Do not include explanations, reasoning, or extra text — output valid JSON only.

Input (JSON array of job details):

""" + json.dumps(job_details) + """

Output JSON schema:

[
  {"jobId": "string", "sponsorship": "Yes" | "No"}
]

Rules:

1. "Yes" if the job description mentions visa sponsorship, work authorization support, or eligibility for international candidates.

2. "No" if the job explicitly requires existing work authorization, citizenship, permanent residency, or does not mention sponsorship.

3. Output must be **pure JSON** — no markdown, no extra keys, no commentary, and no line before or after the JSON array."""
    
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
    job_id = str(row.get("title", "") + "_" + str(row.get("company", "")) + "_" + str(row.get("location", "")))[:100]
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
        return "No"


def add_visa_sponsorship_column(df):
    """
    Add visa sponsorship detection to the dataframe using AWS Bedrock
    """
    print("Analyzing jobs for visa sponsorship...")

    total_count = len(df)

    # Apply visa sponsorship detection to each row
    df["sponsored_job"] = df.apply(detect_visa_sponsorship, axis=1)

    # Count sponsored vs non-sponsored jobs
    sponsored_count = (df["sponsored_job"] == "Yes").sum()

    print("Visa sponsorship analysis complete:")
    print(f"  - Total jobs: {total_count}")
    print(f"  - Sponsored jobs: {sponsored_count}")
    print(f"  - Non-sponsored jobs: {total_count - sponsored_count}")
    print(f"  - Percentage sponsored: {(sponsored_count / total_count) * 100:.1f}%")

    return df


try:
    print("Fetching data from the psql database...")
    df = pd.read_sql(sql_query, engine)
    print(f"Fetched {len(df):,} rows from PostgreSQL database.")
except Exception as e:
    print(f"Error fetching data from PostgreSQL: {e}")
    exit()  # Exit if fetching data from PSQL fails


# Add visa sponsorship detection
df = add_visa_sponsorship_column(df)

# Use .copy() to avoid SettingWithCopyWarning
df_sponsored = df[df["sponsored_job"] == "Yes"].copy()

print(df.columns)

df_sponsored["upload_date"] = df_sponsored["upload_date"].apply(
    lambda x: x.isoformat() if pd.notna(x) else None
)
df_sponsored["date_posted"] = df_sponsored["date_posted"].apply(
    lambda x: x.isoformat() if pd.notna(x) else None
)

try:
    table_name = "job_jobrole_sponsored"
    data_to_insert = df_sponsored.to_dict(orient="records")

    # Convert Timestamp columns to ISO format strings
    response = supabase.table(table_name).insert(data_to_insert).execute()
    print(f"Insert response: {len(response.data)} rows inserted")
except Exception as e:
    print(f"Error inserting data to Supabase: {e}")
    exit()
