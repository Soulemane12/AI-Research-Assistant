# AI Research Assistant

An AI-powered research assistant that leverages Retrieval-Augmented Generation (RAG) to simplify research workflows. This project combines Cortex Search for effective document retrieval, Mistral LLM for text summarization and generation, and Streamlit for an intuitive user interface.

---

## Overview

The AI Research Assistant is built to:
- **Analyze Research Papers**: Upload PDF or text documents and extract key information such as research questions, claims, evidence, and more.
- **Perform Intelligent Searches**: Leverage embeddings and cosine similarity to search and retrieve relevant research papers quickly.
- **Generate Summaries**: Utilize the Mistral LLM for generating concise summaries of text.
- **Provide Relevance Feedback**: Compute similarity metrics to improve search results and provide actionable insights.

The project was built for a hackathon challenge that focused on using Cortex Search, Mistral LLM, and TruLens for enhancing search functionality and research analysis.

---

## Features

- **Research Paper Upload & Analysis**
  - Supports PDF and text file formats.
  - Extracts text from uploaded papers.
  - Generates a summary of the document and key insights (research questions, claims, evidence, and more) using Julep and Mistral LLM.
  
- **Search Functionality**
  - Converts search queries into embeddings.
  - Searches through stored paper embeddings using cosine similarity.
  - Displays search results with relevant metadata and similarity scores.

- **Feedback & Relevance Evaluation**
  - Computes relevance feedback using cosine similarity.
  - Logs processing details and feedback for further improvements.

- **Deployment**
  - Hosted as a live web application using **Streamlit Community Cloud**.

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, Snowflake for data storage and retrieval
- **LLM**: [Mistral LLM](https://www.mistral.ai/) for text generation and summarization
- **Search**: Cortex Search for handling search queries and cosine similarity
- **Evaluation**: TruLens for relevance feedback
- **Other Libraries**: PyMuPDF (fitz) for PDF text extraction, Sentence Transformers for generating text embeddings

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)
- A Snowflake account and required credentials
- API keys for:
  - Julep
  - Mistral
  - (Optional) TruLens

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables:**

    Create a `.env` file in the project root and add the following (fill in your credentials):

    ```env
    JULEP_API_KEY=your_julep_api_key
    SNOWFLAKE_USER=your_snowflake_user
    SNOWFLAKE_PASSWORD=your_snowflake_password
    SNOWFLAKE_ACCOUNT=your_snowflake_account
    SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
    SNOWFLAKE_DATABASE=your_snowflake_database
    SNOWFLAKE_SCHEMA=your_snowflake_schema
    MISTRAL_API_ENDPOINT=your_mistral_api_endpoint
    MISTRAL_API_KEY=your_mistral_api_key
    ```

4. **Run the Application:**

    Start the Streamlit app locally:

    ```bash
    streamlit run app.py
    ```

---

## Project Structure

```plaintext
├── app.py                   # Main Streamlit application
├── utils
│   ├── pdf_extraction.py    # Functions for extracting text from PDFs
│   └── process.py           # Functions to process text, generate summaries, search, and manage embeddings
├── requirements.txt         # List of project dependencies
├── .env                     # Environment variables (do not commit to version control)
└── README.md                # This file


---

## Live Demo

The application is deployed on Streamlit Community Cloud and can be accessed at: [Live Demo URL](https://share.streamlit.io/your-username/your-repository-name)

## Demo Video

Watch a short demo explaining how the AI Research Assistant works: [Demo Video Link](https://www.youtube.com/your-demo-video)

## Challenges & Future Improvements

### Challenges
- Integrating multiple APIs (Julep, Mistral, Snowflake) and managing their error handling.
- Ensuring fast and accurate extraction of text from various file formats.
- Optimizing the search functionality to handle large datasets of research papers.

### Future Improvements
- Enhance the UI/UX with more interactive visualizations.
- Implement additional error logging and alerting mechanisms.
- Expand search capabilities by leveraging additional metadata from research papers.
- Add more fine-tuning options for the LLM and experiment with alternative summarization models.
- Include more robust testing and performance benchmarks.

## Contributing

Feel free to fork this repository and contribute! If you have suggestions, enhancements, or bug fixes, please open an issue or submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- **Snowflake** for their robust data platform.
- **Mistral** and their cutting-edge LLM technology.
- **Streamlit** for making deployment simple and effective.
- **TruLens** for providing insightful evaluation tools.
- The entire hackathon community for inspiring innovative solutions!

