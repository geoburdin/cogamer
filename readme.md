To run the `cogamer` project from the `prepared` branch, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/geoburdin/cogamer.git
   cd cogamer
   git checkout prepared
   ```

2. **Install System-Level Dependencies**:
   ```sh
   brew install portaudio
   ```

3. **Create a Virtual Environment**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate `
   ```

4. **Install Python Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the necessary environment variables (see example file `.env_example`)
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_PROJECT="cogamer"
   VOICE_NAME="Fenrir"
   ```

6. **Run the Agent**:
   ```sh
   python cogamer.py
   ```

7. **Terminate the Agent**:
   Press `Ctrl+C` to stop the agent.

**Note**: It is recommended to use headphones to prevent the agent from detecting and answering its own voice.