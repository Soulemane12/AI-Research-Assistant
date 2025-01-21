import streamlit as st
from utils.pdf_extraction import extract_text_from_pdf
from utils.process import (
    process_with_julep,
    generate_summary,
    search_papers_with_cortex,
    extract_key_sections,
    embed_text,
    upload_data_to_snowflake,
)
import os
import json
from dotenv import load_dotenv
import uuid  # To generate unique IDs for uploaded papers
import warnings
import logging

# ============================
# 1. Setup Logging
# ============================

# Configure logging
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
# 4. Streamlit Page Configuration
# ============================

st.set_page_config(
    page_title="AI-Powered Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# 5. Application Title
# ============================

st.title("AI-Powered Research Assistant")

# ============================
# 6. Sidebar for Navigation
# ============================

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the workflow:", ["Upload a Research Paper", "Search for Relevant Papers"])

# ============================
# 7. Upload a Research Paper Workflow
# ============================

if app_mode == "Upload a Research Paper":
    st.header("Upload and Analyze a Research Paper")

    uploaded_files = st.file_uploader(
        "Upload PDF or Text files", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True
    )

    if st.button("Upload and Analyze"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF or Text file.")
            logging.warning("Upload button clicked without any files uploaded.")
        else:
            st.info("Processing uploaded files...")
            logging.info(f"Processing {len(uploaded_files)} uploaded files.")
            results = []
            processed_papers = []
            for uploaded_file in uploaded_files:
                logging.info(f"Processing file: {uploaded_file.name}")
                # Extract text
                if uploaded_file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = uploaded_file.getvalue().decode('utf-8')
                
                if text:
                    logging.info(f"Text extracted successfully from {uploaded_file.name}.")
                    # Generate summaries and insights
                    summary = generate_summary(text)  # Now uses Mistral
                    julep_result = process_with_julep(text)

                    if isinstance(julep_result, dict) and 'error' not in julep_result:
                        julep_result['filename'] = uploaded_file.name
                        julep_result['summary'] = summary

                        # Create a unique ID and embedding for the paper
                        paper_id = str(uuid.uuid4())
                        embedding = embed_text(text)

                        # Prepare paper data for Snowflake
                        paper_data = {
                            "id": paper_id,
                            "title": julep_result.get("filename", "Unknown Title"),
                            "authors": julep_result.get("authors", "Unknown Authors"),
                            "abstract": julep_result.get("summary", "No summary available"),
                            "embedding_vector": embedding,
                        }
                        processed_papers.append(paper_data)

                        results.append(julep_result)
                        logging.info(f"File {uploaded_file.name} processed and ready for upload.")
                    else:
                        error_message = julep_result.get('error', 'Unknown error occurred.')
                        results.append({
                            "filename": uploaded_file.name,
                            "error": error_message
                        })
                        logging.error(f"Error processing file {uploaded_file.name}: {error_message}")
                else:
                    results.append({
                        "filename": uploaded_file.name,
                        "error": "Failed to extract text."
                    })
                    logging.error(f"Failed to extract text from {uploaded_file.name}.")
            
            # Upload processed papers to Snowflake
            if processed_papers:
                try:
                    upload_data_to_snowflake(processed_papers)
                    st.success("All data uploaded to Snowflake successfully!")
                    logging.info("All processed papers uploaded to Snowflake successfully.")
                except Exception as e:
                    st.error(f"Failed to upload data to Snowflake: {e}")
                    logging.error(f"Failed to upload data to Snowflake: {e}")

            # Display Results
            st.header("Processed Results")
            for result in results:
                st.subheader(result.get('filename'))
                if result.get('error'):
                    st.error(f"Error: {result.get('error')}")
                else:
                    st.markdown(f"**Research Question:** {result.get('research_question')}")
                    st.markdown(f"**Summary:** {result.get('summary')}")
                    st.markdown("**Claims:**")
                    for claim in result.get('claims', []):
                        st.write(f"- {claim}")
                    st.markdown("**Evidence:**")
                    for evidence in result.get('evidence', []):
                        st.write(f"- {evidence}")
                    st.markdown(f"**Claim with Context:** {result.get('claim_with_context')}")
                    st.markdown("**Patterns and Trends:**")
                    st.markdown(result.get('patterns_trends', 'No patterns or trends found.'))
                    st.markdown("**Key Points:**")
                    st.markdown(result.get('key_points', 'No key points found.'))

# ============================
# 8. Search for Relevant Papers Workflow
# ============================

elif app_mode == "Search for Relevant Papers":
    st.header("Search for Relevant Papers")
    
    # Text input for the search query
    search_query = st.text_input(
        "Enter keywords, topics, or research questions", 
        help="Type keywords to find papers."
    )
    
    # Button to trigger the search
    if st.button("Search"):
        if not search_query:
            st.warning("Please enter a search query.")
            logging.warning("Search button clicked without any query entered.")
        else:
            st.info("Searching for relevant papers...")
            logging.info(f"Performing search with query: {search_query}")
            try:
                # Generate embedding for the search query
                query_embedding = embed_text(search_query)
                
                if not query_embedding:
                    st.error("Failed to generate embedding for the query.")
                    logging.error("Embedding generation failed for the search query.")
                else:
                    # Search papers using the embedding
                    search_results = search_papers_with_cortex(query_embedding)
                    
                    if not search_results:
                        st.warning("No relevant papers found.")
                        logging.info("No relevant papers found for the query.")
                    else:
                        for paper in search_results:
                            st.subheader(paper.get('title'))
                            st.markdown(f"**Authors:** {paper.get('authors')}")
                            st.markdown(f"**Similarity:** {paper.get('similarity'):.4f}")
                            st.markdown(f"**Abstract:** {paper.get('abstract')}")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                logging.error(f"Search failed with error: {e}")