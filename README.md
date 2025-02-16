# Student Assistance Chatbot

## Overview
The **Student Assistance Chatbot** is an AI-powered chatbot that helps students by answering queries related to their studies, career opportunities, and educational resources. It supports both **text-based and voice-based interactions** and features a **web interface built using Streamlit**.

## Features
- **Conversational AI** using **LangChain** for natural language understanding.
- **Voice Assistance** for both speech-to-text and text-to-speech functionalities.
- **Web Application** built with **Streamlit** for an interactive user experience.
- **Efficient Query Retrieval** using **Pinecone** for semantic search.
- **Integration with Large Language Models (LLMs)** for intelligent responses.
- **PDF Processing** to extract and analyze content from documents.
- **Environment Configuration** using **dotenv** for API keys and configurations.

## Tech Stack
The chatbot is built using the following libraries and technologies:

| Library               | Purpose                                      |
|-----------------------|----------------------------------------------|
| **LangChain**        | Framework for building conversational AI    |
| **langchain_community** | Community integrations for LangChain        |
| **pinecone-client**  | Vector database for efficient search        |
| **streamlit**        | Web framework for interactive UI            |
| **python-dotenv**    | Environment variable management             |
| **google-generativeai** | Google AI models for responses              |
| **pypdf**            | PDF processing and content extraction       |
| **transformers**     | Pretrained models for NLP tasks             |
| **nltk**             | Natural Language Toolkit for text processing |
| **langchain_huggingface** | Integration with Hugging Face models         |
| **speechrecognition** | Converts voice input to text               |
| **pyttsx3**          | Text-to-speech engine                       |

## Installation
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/CodeMosaic7/Student-Assistance-Chatbot.git
   cd Student-Assistance-Chatbot
   ```
2. **Create a Virtual Environment and Activate It:**
   ```sh
   python -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   myenv\Scripts\activate     # On Windows
   ```
3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables:**
   - Create a `.env` file in the root directory and add the required API keys (e.g., OpenAI API Key, Pinecone API Key, etc.).

## Usage
- **Run the Chatbot Locally:**
  ```sh
  streamlit run app.py
  ```
- **Interacting with the Chatbot:**
  - Type your queries in the chat input.
  - Use voice input for speech recognition.
  - Receive both text and voice responses.

## Contributing
We welcome contributions! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.



