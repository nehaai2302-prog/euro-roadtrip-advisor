# EuroRoad Advisor

A Streamlit-based AI travel assistant for European road trips, powered by RAG (Retrieval-Augmented Generation) and custom tools for weather, fuel costs, and toll calculations.

## Features

- **Interactive Chat Interface**: Ask questions about European driving laws, routes, and travel tips
- **Personalized Recommendations**: Tailor advice based on traveler type (solo, family, elderly) and trip style (balanced, scenic, budget)
- **RAG-Powered Knowledge Base**: Retrieves information from a curated knowledge base of European travel documents
- **Tool Integration**: 
  - Weather forecasts for destinations
  - Fuel cost estimates
  - Toll/vignette calculations
- **Safety Layer**: Filters inappropriate or off-topic queries
- **Usage Tracking**: Monitors API token usage and estimated costs
- **LangSmith Tracing**: Logs RAG requests, tool calls, and agent flow for monitoring and debugging
- **Session Management**: Persistent chat history and preferences

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Conda or virtual environment (recommended)

### Setup Steps

1. **Clone the repository and navigate to the project directory**:
   ```bash
   git clone https://github.com/nehaai2302-prog/euroroad-advisor.git
   cd euroroad-advisor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   conda create -n euroroad python=3.9
   conda activate euroroad
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Prepare the knowledge base** (if not already done):
   Run the ingestion script to process travel documents:
   ```bash
   python ingest.py
   ```

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the displayed URL (usually http://localhost:8501)

### Using the App

1. **Configure your trip** in the sidebar:
   - Select traveler type
   - Choose trip style
   - Set preferences (avoid tolls, highways, etc.)

2. **Chat with the AI**:
   - Ask questions like "What's the best route from Paris to Rome?"
   - "How do I handle tolls in Germany?"
   - "What's the weather like in Barcelona?"

3. **View explanations**:
   - Expand the "System Logic" section to see which tools were used and knowledge chunks retrieved

### Example Queries

- "Plan a scenic route from London to Edinburgh"
- "How much fuel will I need for a 500km trip?"
- "What's the toll cost for driving in France?"
- "Are there any special driving rules in Scandinavia?"

## Project Structure

```
project2/
├── app.py                    # Main Streamlit application
├── rag_pipeline_fast.py      # RAG pipeline with agent logic
├── tools.py                  # Custom tools (weather, fuel, toll)
├── ingest.py                 # Data ingestion script for knowledge base
├── safety.py                 # Query safety and filtering
├── data/                     # Knowledge base documents
│   ├── europe_driving_master.txt
│   ├── scandinavia_driving_rules.txt
│   ├── traveller_profiles.txt
│   └── ...
├── chroma_db/                # Vector database storage
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
└── README.md                 # This file
```
## Mermaid Diagram (System Architecture)

flowchart TD

    A[User] --> B[Streamlit UI - app.py]

    B --> C[Safety Layer - check_safety]

    C -->|Safe Query| D[RAG Pipeline - rag_pipeline_fast.py]

    D --> E[Intent Router]

    E -->|weather| F[Weather Tool]
    E -->|fuel| G[Fuel Cost Tool]
    E -->|toll| H[Toll Tool]
    E -->|itinerary| I[LLM Itinerary Generator]
    E -->|chitchat| J[LLM Chitchat Handler]
    E -->|rag| K[Retriever - ChromaDB]

    K --> L[Knowledge Base Documents]

    F --> M[Context Builder]
    G --> M
    H --> M
    K --> M
    I --> M
    J --> M

    M --> N[Final LLM Response]

    N --> O[Response + Steps + Token Usage]

    O --> B

    B --> P[Chat UI + History + Export]

    subgraph Tools
        F
        G
        H
    end

    subgraph Data Layer
        L
    end

    subgraph Observability
        Q[LangSmith Tracing]
    end

    D --> Q
    M --> Q
    N --> Q

  

## RAG Pipeline


flowchart LR

    A[User Query] --> B[Embedding Model]

    B --> C[Vector Search - ChromaDB]

    C --> D[Top-K Chunks]

    D --> E[Context Injection]

    E --> F[LLM]

    F --> G[Final Answer]


## Intent Routing Logic

flowchart TD

    A[User Query] --> B[Intent Classifier]

    B -->|Weather Query| C[Weather Tool]
    B -->|Fuel Query| D[Fuel Tool]
    B -->|Toll Query| E[Toll Tool]
    B -->|Trip Planning| F[Itinerary Generator]
    B -->|General Chat| G[Chitchat Handler]
    B -->|Driving Laws| H[RAG Retrieval]

    C --> I[Final Answer]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I

## Key Components

### app.py
- Streamlit UI with sidebar controls
- Chat interface and message history
- Session state management
- Integration with RAG pipeline

### rag_pipeline_fast.py
- Intent classification (itinerary, weather, fuel, toll, chitchat, RAG)
- Tool routing for weather/fuel/toll, direct LLM handling for itinerary/chitchat
- RAG retrieval from ChromaDB
- Final answer generation

### tools.py
- `get_weather_forecast()`: Weather API integration
- `estimate_fuel_cost()`: Fuel cost calculations
- `calculate_toll_vignette()`: Toll fee estimates

### ingest.py
- Processes text documents from `data/` folder
- Creates vector embeddings
- Stores in ChromaDB for retrieval

### safety.py
- `check_safety()`: Filters unsafe or off-topic queries

### data/ (Knowledge Base)
- Curated travel documents covering:
  - Driving laws and regulations
  - Route information
  - Cultural tips
  - Safety guidelines

## Dependencies

Key packages (see `requirements.txt` for full list):
- `streamlit`: Web UI framework
- `langchain`: LLM orchestration
- `chromadb`: Vector database
- `openai`: OpenAI API client
- `langsmith`: RAG tracing, logging, and monitoring
- `python-dotenv`: Environment variable management
- `requests`: HTTP client for APIs

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration
The app uses GPT-3.5-turbo by default. To change models, modify `rag_pipeline_fast.py`.

### Knowledge Base Updates
To add new documents:
1. Place text files in `data/` folder
2. Run `python ingest.py` to re-index

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure `.env` file exists with correct API key
   - Check file is in project root

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Ensure you're in the correct virtual environment

3. **Knowledge base not working**
   - Run `python ingest.py` to initialize ChromaDB
   - Check `data/` folder has documents

4. **Streamlit not starting**
   - Ensure port 8501 is available
   - Try `streamlit run app.py --server.port 8502`

### Debug Mode
Add `st.write()` statements in `app.py` for debugging, or check terminal output for errors.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 Python style guide
- Add docstrings to functions
- Test new features with various queries

## License

This project is for educational purposes. Please respect OpenAI's terms of service and any data usage policies.

## Future Enhancements

- [ ] Multi-language support
- [ ] Real-time traffic data integration
- [ ] User account system for saved trips
- [ ] Mobile app version
- [ ] Integration with Google Maps API
- [ ] Voice input/output
