import os
from sqlalchemy import create_engine
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

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

# Exception handling for fetching max upload date
try:
    response = (
        supabase.table("job_jobrole_sponsored")
        .select("upload_date")
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
    j."rawText" AS raw_text,
    j."datePosted" AS date_posted,
    j."yearsExpRequired" AS years_exp_required,
    j."uploadDate" AS upload_date
FROM "karmafy_job" j
LEFT JOIN "karmafy_jobrole" jr
       ON j."roleId"::bigint = jr.id
WHERE j."uploadDate" > '{max_upload_date}'
  AND j."uploadDate" <= CURRENT_TIMESTAMP;
"""


def detect_visa_sponsorship(text):
    if pd.isna(text) or text == "":
        return "No"

    # Convert to lowercase for case-insensitive matching
    text_lower = str(text).lower()

    # Keywords that indicate visa sponsorship
    visa_keywords = [
        # General sponsorship keywords
        "sponsorship available",
        "visa sponsorship available",
        "h-1b sponsorship available",
        "will sponsor",
        "sponsorship provided",
        "sponsorship offered",
        "sponsorship opportunities",
        # Open to sponsoring
        "open to sponsoring",
        "employer will sponsor visas",
        "company sponsorship available",
        "visa support provided",
        "eligible for visa sponsorship",
        # Additional visa sponsorship keywords
        "may provide visa sponsorship for certain positions",
        "visa sponsorship available for this position",
        "we do sponsor visas",
        "employment sponsorship offered: yes",
        "all visa requests will be discussed on a case by case basis to determine if we can sponsor",
        "visa sponsorship and relocation stipend to bring you to sf, if possible",
        "will consider sponsoring a new qualified applicant for employment authorization for this position",
        "visa sponsorship decisions will be made on a case-by-case basis",
        "sponsorship available",
        "visa sponsorship may be offered for this role",
        "may offer employer visa sponsorship to applicants",
        "we are open to sponsoring candidates currently in the u.s. who need to transfer their active h1-b visa",
        # Extended visa sponsorship keywords
        "sponsorship through the h-1b lottery",
        "visa sponsorship: we do sponsor visas! however, we aren't able to successfully sponsor visas for every role and every candidate. but if we make you an offer, we will make every reasonable effort to get you a visa, and we retain an immigration lawyer to help with this",
        "work visas - all visa requests will be discussed on a case by case basis to determine if we can sponsor",
        "visa sponsorship to bring you",
        "sponsorship: we are open to sponsoring candidates currently in the u.s. who need to transfer their active h1-b visa",
        "supports visa sponsorship, sponsorship opportunities may be limited to certain roles and skills",
        "visa sponsorship: we do sponsor visas! however, we aren't able to successfully sponsor visas for every role and every candidate",
        "visa sponsorship is available for this position",
        "we provide visa sponsorship for candidates selected for this role",
        "will sponsor foreign nationals for work visas",
        "visa sponsorship - available for those who would like to relocate to the us after being hired",
        "visa sponsorship available",
        "we will consider applicants requiring sponsorship for this opportunity",
        "visa sponsorship & assistance: we support your visa process and guide you through all the paperwork",
        "consideration for work authorization sponsorship",
        "visa sponsorship - we offer visa sponsorship for eligible employees",
        "visa sponsorship offered",
        "our client will sponsor h1-b's",
        "h1b sponsorship",
        "visa sponsorship available: yes",
        "sponsorship available: yes",
        "occasionally offers work authorization sponsorship for critical need roles",
        "work authorization sponsorship is available for this position",
        "visa sponsorship: we are able to provide employment visa sponsorship for qualified candidates",
        "f-1 visa sponsorship available",
        "j-1 and f-1 visa sponsorship available",
        "h-1b sponsorship available",
        "tn visa sponsorship is available",
        # AI-words
        "willing to sponsor",
        "able to sponsor",
        "can sponsor",
        "we sponsor",
        "h1b visa sponsorship",
        "h-1b visa sponsorship",
        "h1-b sponsorship",
        "opt sponsorship",
        "opt extension sponsorship",
        "stem opt sponsorship",
        "cpt sponsorship",
        "f1 opt sponsorship",
        "f-1 opt sponsorship",
        "green card sponsorship",
        "green card sponsorship available",
        "immigration sponsorship",
        "immigration support",
        "immigration assistance",
        "work permit sponsorship",
        "work authorization sponsorship available",
        "visa transfer support",
        "h1b transfer",
        "h-1b transfer",
        "e-3 visa sponsorship",
        "l-1 visa sponsorship",
        "o-1 visa sponsorship",
        "tn visa sponsorship",
        "tn status sponsorship",
        "sponsor work visas",
        "sponsors h1b",
        "sponsors h-1b",
        "visa sponsorship for qualified candidates",
        "visa sponsorship for international candidates",
        "open to visa sponsorship",
        "provides visa sponsorship",
        "offers visa sponsorship",
        "visa sponsorship program",
        "sponsorship on a case-by-case basis",
        "may sponsor the right candidate",
        "will sponsor qualified candidates",
        "visa assistance available",
        "relocation and visa support",
        "visa sponsorship for exceptional candidates",
        "h-1b cap-exempt sponsorship",
        "cap-exempt h1b sponsorship",
        "employment-based visa sponsorship",
        "sponsorship for foreign nationals",
        "we can sponsor your visa",
        "company will sponsor",
        "visa sponsorship considered",
        "visa sponsorship possible",
        "willing to provide sponsorship",
        "able to provide sponsorship",
        "sponsorship available for this role",
        "visa sponsorship for the right fit",
        "we handle visa sponsorship",
        "full visa sponsorship support",
        "comprehensive visa sponsorship",
        "employer-sponsored work visa",
        "visa petition support",
        "will file for h-1b",
        "gc sponsorship",
        "perm sponsorship",
        "sponsorship after probation",
        "sponsorship after trial period",
        # extra
        "h1-b visa sponsorship",
        "employment visa sponsorship",
        "work visa sponsorship",
        "temporary work visa",
        "nonimmigrant visa",
        "legal sponsorship",
        "petition for visa",
        "labor certification",
        "prevailing wage",
        "visa processing",
        "immigration processing",
        "sponsor h1b",
        "sponsor h-1b",
        "h1b cap",
        "h-1b cap",
        "h1b lottery",
        "h-1b lottery",
        "cap exempt",
        "cap-exempt",
        "h1b transfer sponsorship",
        "h-1b transfer sponsorship",
        "new h1b",
        "new h-1b",
        "consular processing",
        "change of status",
        "visa stamping",
        "ds-2019",
        "i-129",
        "lca",
        "labor condition application",
        "immigration attorney",
        "immigration lawyer",
        "visa attorney",
        "sponsorship for work authorization",
        "ead sponsorship",
        "stem extension",
        "cap gap extension",
        "h1b visa transfer",
        "h-1b visa transfer",
        "visa petition",
        "immigration petition",
        "sponsor employment",
        "employment sponsorship",
        "job sponsorship",
        "professional visa",
        "skilled worker visa",
        "specialty occupation",
        "specialty occupations",
        "bachelor's degree requirement",
        "degree requirement for visa",
        "visa eligible position",
        "sponsorship eligible",
        "can sponsor visas",
        "provides sponsorship",
        "offers sponsorship",
        "gives sponsorship",
        "employer provides sponsorship",
        "immigration benefits",
        "relocation package includes visa",
        "international relocation support",
        "global mobility",
        "talent mobility",
        "immigration support services",
        "visa and immigration assistance",
        "work permit assistance",
        "legal work status sponsorship",
        "authorization sponsorship",
        "foreign talent welcome",
        "international hires welcome",
        "global talent acquisition",
        "visa for right candidate",
        "immigration for talented professionals",
        "sponsor for exceptional candidates",
        "h1b candidates welcome",
        "h-1b applicants welcome",
        "accepting h1b applications",
        "considering h1b transfers",
        "open to h1b candidates",
        "h1b friendly",
        "h-1b friendly",
        "visa friendly employer",
        "sponsorship friendly",
        "supports h1b visa",
        "supports h-1b visa",
        "h1b visa support",
        "h-1b visa support",
        "l1 visa sponsorship",
        "l-1 visa sponsorship",
        "e3 visa sponsorship",
        "o1 visa sponsorship",
        "j1 visa sponsorship",
        "f1 visa sponsorship",
        "h1b1 visa sponsorship",
        "e1 visa sponsorship",
        "e2 visa sponsorship",
    ]

    # Check for negative indicators (jobs that explicitly don't sponsor)
    negative_keywords = [
        "sponsorship available no",
        "no visa sponsorship",
        "does not sponsor",
        "will not sponsor",
        "no h1b",
        "no h-1b",
        "no work visa",
        "us citizens only",
        "green card required",
        "must have work authorization",
        "no sponsorship",
        "sponsorship not available",
        "visa sponsorship available: not available",
        "visa sponsorship available: no",
        "not eligible without sponsorship",
        "this role is not eligible for visa sponsorship or relocation assistance",
        "this position is unable to provide work authorization sponsorship",
        "candidates for this position must be authorized to work in the united states and not require work authorization sponsorship by our company for this position",
        "this position is ineligible for visa sponsorship",
        "this position is ineligible for employment visa sponsorship",
        "this role is not eligible for visa sponsorship",
        "employment sponsorship offered: no",
        "this position is not eligible for visa sponsorship",
        # AI-words
        "cannot sponsor",
        "unable to sponsor",
        "will not provide sponsorship",
        "does not provide sponsorship",
        "no h1-b sponsorship",
        "no h1b sponsorship",
        "h1b not available",
        "h-1b not available",
        "must be authorized to work in the us",
        "must be authorized to work in the united states",
        "must be legally authorized to work",
        "must be authorized to work without sponsorship",
        "us work authorization required",
        "valid us work authorization required",
        "must have us work authorization",
        "work authorization required",
        "only us citizens",
        "us citizens and green card holders only",
        "us citizens and permanent residents only",
        "permanent residents only",
        "green card holders only",
        "citizenship required",
        "us citizenship required",
        "must be a us citizen or permanent resident",
        "citizen or green card holder",
        "citizen or permanent resident",
        "ead required",
        "must have ead",
        "employment authorization document required",
        "gc or us citizen only",
        "gc or citizen only",
        "sponsorship is not available",
        "we do not sponsor",
        "we cannot sponsor",
        "we are unable to sponsor",
        "company does not sponsor",
        "employer does not sponsor",
        "not offering sponsorship",
        "no sponsorship provided",
        "no visa support",
        "visa support not available",
        "we do not provide visa sponsorship",
        "sponsorship will not be provided",
        "no immigration sponsorship",
        "immigration sponsorship not available",
        "must currently have work authorization",
        "existing work authorization required",
        "current us work authorization required",
        "no opt sponsorship",
        "no cpt sponsorship",
        "not eligible for sponsorship",
        "ineligible for sponsorship",
        "does not offer sponsorship",
        "unable to provide sponsorship",
        "we cannot provide work authorization",
        "authorized to work in us without sponsorship",
        "must have unrestricted work authorization",
        "unrestricted right to work in the us",
        "no relocation or visa sponsorship",
        "visa sponsorship is not offered",
        "this position does not include visa sponsorship",
        "not open to visa sponsorship",
        "seeking candidates who do not require sponsorship",
        "candidates must not require sponsorship",
        "no third-party or visa sponsorship",
        "must possess us work authorization",
        "proof of eligibility to work required",
        # extra
        "no h1-b",
        "no h1b visa",
        "no h-1b visa",
        "no work authorization sponsorship",
        "not able to sponsor",
        "sponsorship not provided",
        "sponsorship not offered",
        "no employment sponsorship",
        "no job sponsorship",
        "no immigration sponsorship",
        "no legal sponsorship",
        "visa sponsorship not provided",
        "work visa not provided",
        "must have current work authorization",
        "must be authorized to work",
        "employment authorization required",
        "us employment authorization required",
        "must have us work permit",
        "us work permit required",
        "permanent residency required",
        "green card holder required",
        "us person required",
        "local candidates only",
        "no relocation sponsorship",
        "no visa assistance",
        "no immigration assistance",
        "no legal assistance for visa",
        "not providing sponsorship",
        "sponsorship unavailable",
        "visa sponsorship unavailable",
        "work authorization sponsorship not available",
        "employment sponsorship not available",
        "does not support h1b",
        "does not support h-1b",
        "h1b not supported",
        "h-1b not supported",
        "no h1b transfers",
        "no h-1b transfers",
        "not accepting h1b candidates",
        "not considering h1b applications",
        "h1b applicants not accepted",
        "h-1b applicants not accepted",
        "only us citizens or permanent residents",
        "must be us citizen or permanent resident",
        "no visa candidates",
        "no foreign candidates",
        "local hiring only",
        "no international candidates",
        "domestic candidates only",
        "no sponsorship for this role",
        "role does not offer sponsorship",
        "position does not offer sponsorship",
        "sponsorship is not available for this position",
        "we do not sponsor visas",
        "organization does not sponsor",
        "no visa sponsorship available",
        "no work visa sponsorship",
        "no employment visa sponsorship",
        "no temporary visa sponsorship",
        "non-immigrant visa not sponsored",
        "no non-immigrant visa sponsorship",
        "must have existing work authorization",
        "valid work authorization required",
        "must possess work authorization",
        "work authorization must be current",
        "no new work authorization",
        "no initial work authorization",
        "not for opt candidates",
        "no opt",
        "no cpt",
        "not accepting opt",
        "not considering cpt",
        "no stem opt",
        "no practical training",
        "must have work visa already",
        "existing work visa required",
        "transfer not available",
        "no visa transfers",
        "sponsorship: no",
        "visa sponsorship: no",
        "work authorization: must have",
        "authorization: required",
        "no l1 sponsorship",
        "no l-1 sponsorship",
        "no tn visa",
        "no e3 visa",
        "no o1 visa",
        "no j1 sponsorship",
        "no f1 sponsorship",
        "all candidates must have work authorization",
        "authorization to work in us is required",
        "must be legally authorized to work in the united states",
        "we cannot sponsor work visas at this time",
        "unable to provide visa sponsorship",
        "not eligible for work visa sponsorship",
        "this role does not qualify for visa sponsorship",
        "position ineligible for any visa sponsorship",
        "no exceptions for work authorization",
        "strictly no visa sponsorship",
        "absolutely no sponsorship",
        "sponsorship is not an option",
        "visa sponsorship explicitly not available",
    ]
    # First check for negative indicators
    for neg_keyword in negative_keywords:
        if neg_keyword in text_lower:
            return "No"

    # Then check for positive indicators
    for keyword in visa_keywords:
        if keyword in text_lower:
            return "Yes"

    return "No"


def add_visa_sponsorship_column(df):
    """
    Add visa sponsorship detection to the dataframe
    """
    print("Analyzing jobs for visa sponsorship...")

    # Combine title and description for analysis
    df["combined_text"] = (
        df["title"].fillna("") + " " + df.get("description", "").fillna("")
    )

    # Apply visa sponsorship detection
    df["sponsored_job"] = df["combined_text"].apply(detect_visa_sponsorship)

    # Remove the temporary combined_text column
    df = df.drop("combined_text", axis=1)

    # Count sponsored vs non-sponsored jobs
    sponsored_count = (df["sponsored_job"] == "Yes").sum()
    total_count = len(df)

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

df_sponsored = df[df["sponsored_job"] == "Yes"]

print(df.columns)

df_sponsored["upload_date"] = df["upload_date"].apply(
    lambda x: x.isoformat() if pd.notna(x) else None
)
df_sponsored["date_posted"] = df["date_posted"].apply(
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
