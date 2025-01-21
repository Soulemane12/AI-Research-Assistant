import os
import json
import time
import uuid
import yaml
import warnings
import logging
from dotenv import load_dotenv
from julep import Julep
from sentence_transformers import SentenceTransformer
import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from mistralai import Mistral  # Import Mistral client

# ============================
# 1. Setup Logging
# ============================

# Configure logging at the beginning of your script
logging.basicConfig(
    filename='app.log',  # Log file name
    filemode='a',         # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO    # Log level
)

# ============================
# 2. Suppress PyTorch Warnings
# ============================

# Suppress the PyTorch torch.classes warning if it's non-critical
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# ============================
# 3. Load Environment Variables
# ============================

load_dotenv()

# ============================
# 4. Initialize Julep Client
# ============================

JULEP_API_KEY = os.getenv('JULEP_API_KEY')
if not JULEP_API_KEY:
    logging.error("JULEP_API_KEY not found in environment variables.")
    raise EnvironmentError("JULEP_API_KEY not found in environment variables.")

julep_client = Julep(api_key=JULEP_API_KEY)

# ============================
# 5. Initialize Snowflake Connection Details
# ============================

SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# Validate environment variables
required_env_vars = [
    'SNOWFLAKE_USER',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_WAREHOUSE',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_SCHEMA',
    'MISTRAL_API_ENDPOINT',  # Ensure Mistral variables are present
    'MISTRAL_API_KEY'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ============================
# 6. Load Sentence Transformer Model
# ============================

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("SentenceTransformer model loaded successfully.")

# ============================
# 7. Initialize Mistral Client
# ============================

MISTRAL_API_ENDPOINT = os.getenv('MISTRAL_API_ENDPOINT')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    logging.info("Mistral client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Mistral client: {e}")
    raise

# ============================
# 8. Snowflake Connection Function
# ============================

def create_snowflake_connection():
    """Establishes a connection to Snowflake."""
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        logging.info("Snowflake connection established successfully.")
        return conn
    except Exception as e:
        logging.error(f"Snowflake connection error: {e}")
        return None

# ============================
# 9. Search Papers Function
# ============================

def search_papers_with_cortex(query_embedding):
    """Searches for relevant papers by computing cosine similarity."""
    conn = create_snowflake_connection()
    if not conn:
        logging.error("Failed to establish Snowflake connection.")
        return []
    
    try:
        cursor = conn.cursor()
        
        # Ensure the embedding vector is a list of floats
        if not isinstance(query_embedding, list):
            raise ValueError("query_embedding must be a list of floats.")
        
        # Convert the embedding vector to a comma-separated string
        array_elements = ', '.join(map(str, query_embedding))
        
        # Use COSINE_SIMILARITY to find similar papers with ARRAY_CONSTRUCT
        sql = f"""
            SELECT 
                ID,
                TITLE,
                AUTHORS,
                ABSTRACT,
                COSINE_SIMILARITY(EMBEDDING_VECTOR, ARRAY_CONSTRUCT({array_elements})) AS similarity
            FROM HACATHON.PUBLIC.RESEARCH_PAPERS
            WHERE COSINE_SIMILARITY(EMBEDDING_VECTOR, ARRAY_CONSTRUCT({array_elements})) IS NOT NULL
            ORDER BY similarity DESC
            LIMIT 10;
        """
        
        # Debugging: Log the final SQL query
        logging.info(f"Executing SQL Query:\n{sql}")
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        papers = []
        for row in results:
            papers.append({
                "id": row[0],
                "title": row[1],
                "authors": row[2],
                "abstract": row[3],
                "similarity": row[4]
            })
        logging.info(f"Found {len(papers)} relevant papers.")
        return papers

    except Exception as e:
        logging.error(f"Error searching papers: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ============================
# 10. Julep Agent and Task Creation
# ============================

def create_julep_agent():
    """Creates a Julep agent for processing."""
    try:
        logging.info("Creating Julep agent...")
        agent = julep_client.agents.create(
            name="Research Insights Agent",
            model="o1-preview",
            about="Extracts research questions, claims, evidence, and contextualizes claims from research papers."
        )
        logging.info("Julep agent created successfully.")
        return agent
    except Exception as e:
        logging.error(f"Error creating Julep agent: {e}")
        raise

def create_julep_task(agent_id):
    """Creates a Julep task."""
    try:
        logging.info("Creating Julep task...")
        task_yaml = """
name: Research Extractor
description: Extract key insights from a research paper, including research question, claims, evidence, contextualized claims, highlight patterns or trends, and key points and findings.

main:
  - prompt:
      - role: system
        content: You are an AI specialized in analyzing research papers.
      - role: user
        content: >
          Please analyze the following research paper text and extract:
          1. The main research question.
          2. Key claims made in the paper.
          3. Evidence supporting those claims.
          4. Provide a claim along with its context.
          5. Highlight patterns, trends, or relationships within the data.
          6. Key points and findings

          Text:
          {{_.research_text}}

          Return the results in the following JSON structure:
          ```json
          {
            "research_question": "<string>",
            "claims": ["<string>", "<string>"],
            "evidence": ["<string>", "<string>"],
            "claim_with_context": "<string>",
            "patterns_trends": "<string>",
            "key_points": "<string>"
          }
          ```
    unwrap: true
"""
        task = julep_client.tasks.create(
            agent_id=agent_id,
            **yaml.safe_load(task_yaml)
        )
        logging.info("Julep task created successfully.")
        return task
    except Exception as e:
        logging.error(f"Error creating Julep task: {e}")
        raise

# ============================
# 11. Initialize Julep Agent and Task
# ============================

try:
    agent = create_julep_agent()
    task = create_julep_task(agent.id)
except Exception as e:
    logging.error(f"Failed to initialize Julep agent and task: {e}")
    raise

# ============================
# 12. Process Text with Julep
# ============================

def process_with_julep(text):
    """Processes text with Julep API to extract research insights."""
    logging.info("Starting Julep task execution...")
    try:
        execution = julep_client.executions.create(
            task_id=task.id,
            input={"research_text": text}
        )

        # Wait for the execution to complete
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ['succeeded', 'failed']:
                break
            logging.info(f"Current status: {result.status}... waiting.")
            time.sleep(2)

        if result.status == 'succeeded':
            logging.info("Task succeeded. Raw output received.")
            try:
                # Clean and parse JSON output
                output_str = result.output.strip()
                if output_str.startswith('```json'):
                    output_str = output_str[7:].strip()
                if output_str.endswith('```'):
                    output_str = output_str[:-3].strip()
                parsed_result = json.loads(output_str)
                logging.info("Successfully parsed JSON from Julep output.")
                return parsed_result
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                logging.error(f"Raw output that failed to parse: {result.output}")
                return {"error": "Failed to parse response."}
        else:
            logging.error(f"Task failed with error: {result.error}")
            return {"error": str(result.error)}
    except yaml.YAMLError as e:
        logging.error(f"YAML formatting error: {e}")
        return {"error": "YAML formatting error occurred."}
    except Exception as e:
        logging.error(f"Unexpected error in process_with_julep: {str(e)}")
        return {"error": "Unexpected error occurred."}

# ============================
# 13. Generate Summary Using Mistral
# ============================

def generate_summary_with_mistral(text):
    """Generates a summary of the text using Mistral LLM."""
    logging.info("Generating summary with Mistral...")
    try:
        prompt = f"Summarize the following text in less than 100 words:\n\n{text}"
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.5,  # Adjust as needed for creativity
            max_tokens=150      # Adjust based on desired summary length
        )
        summary = chat_response.choices[0].message.content.strip()
        logging.info("Summary generated successfully with Mistral.")
        return summary
    except Exception as e:
        logging.error(f"Error generating summary with Mistral: {e}")
        return "An error occurred while generating the summary. Please try again later."

def generate_summary(text):
    """Generates a summary of the text using Mistral."""
    return generate_summary_with_mistral(text)  # Use Mistral for summarization

# ============================
# 14. Extract Key Sections Using Julep
# ============================

def extract_key_sections(text):
    """Extracts specific sections from the text using Julep API."""
    try:
        # Properly escape the text to avoid YAML issues
        escaped_text = text.replace('"', '\\"').replace("'", "\\'")
        task_yaml = f"""
name: Section Extraction Task
description: Extract specific sections from the research paper.

main:
  - prompt:
      - role: system
        content: "Extract the methodology, results, and discussion sections from the following text: {escaped_text}"
      - role: user
        content: >
          Please extract the following sections from the text:
          1. Methodology
          2. Results
          3. Discussion
    unwrap: true
"""
        task = julep_client.tasks.create(agent_id=agent.id, **yaml.safe_load(task_yaml))
        execution = julep_client.executions.create(task_id=task.id, input={"text": text})

        # Wait for the execution to complete
        while True:
            result = julep_client.executions.get(execution.id)
            if result.status in ['succeeded', 'failed']:
                break

            logging.info(f"Current status: {result.status}... waiting.")
            time.sleep(2)

        if result.status == "succeeded":
            return result.output  # Return the extracted sections
        else:
            logging.error(f"Failed to extract sections: {result.error}")
            return {"error": str(result.error)}

    except yaml.YAMLError as e:
        logging.error(f"YAML formatting error: {e}")
        return {"error": "YAML formatting error occurred."}
    except Exception as e:
        logging.error(f"Unexpected error in extract_key_sections: {str(e)}")
        return {"error": "Unexpected error occurred."}

# ============================
# 15. Generate Embedding
# ============================

def embed_text(text):
    """Generates an embedding for the given text using Sentence Transformers."""
    try:
        embedding = embedding_model.encode(text).tolist()  # Convert to list for easier handling
        logging.info("Text embedding generated successfully.")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return []

# ============================
# 16. Upload Data to Snowflake
# ============================

def upload_data_to_snowflake(papers):
    """Uploads research paper data to Snowflake using write_pandas."""
    conn = create_snowflake_connection()
    if not conn:
        logging.error("Failed to establish Snowflake connection for uploading data.")
        return

    try:
        # Create DataFrame
        df = pd.DataFrame(papers)

        # Ensure column order matches table
        expected_columns = ['id', 'title', 'authors', 'abstract', 'embedding_vector']
        if not all(col in df.columns for col in expected_columns):
            logging.error(f"DataFrame missing required columns. Expected columns: {expected_columns}")
            return

        df = df[expected_columns]

        # Ensure 'embedding_vector' is treated as a list
        df['embedding_vector'] = df['embedding_vector'].apply(lambda x: x if isinstance(x, list) else [])

        # Use write_pandas to upload
        success, nchunks, nrows, _ = write_pandas(conn, df, 'RESEARCH_PAPERS', quote_identifiers=False)
        if success:
            logging.info("Data uploaded to Snowflake successfully.")
        else:
            logging.error("Failed to upload data to Snowflake.")
    except Exception as e:
        logging.error(f"Error uploading data to Snowflake: {e}")
    finally:
        conn.close()
