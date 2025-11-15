# 🎙️ Voice Agent with RAG + MCP Integration

A complete voice-enabled AI agent that combines:
- **Speech-to-Text** (Whisper) for voice input
- **RAG** (Your existing LangChain pipeline) for knowledge retrieval
- **MCP Tools** for external actions (weather, Wikipedia, etc.)
- **LLM Orchestration** (Ollama) for intelligent decision-making
- **Text-to-Speech** (pyttsx3) for voice responses

## 📁 Project Structure

```
langchain-rag-project/
├── data/
│   └── sample_docs/
├── vectorstore/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── embeddings_manager.py
│   ├── vectorstore_manager.py
│   ├── retriever_manager.py
│   ├── llm_manager.py
│   ├── rag_chain.py           # Your existing RAG pipeline
│   └── evaluator.py
├── voice_agent.py              # NEW: Main voice agent
├── stt.py                      # NEW: Speech-to-text
├── tts.py                      # NEW: Text-to-speech
├── mcp_tools.py                # NEW: MCP tools integration
├── orchestrator.py             # NEW: LLM orchestration
├── main.py
├── requirements.txt            # UPDATED: Added voice agent deps
└── README.md
```

## 🚀 Quick Start

python3 -m venv venv

source venv/bin/activate                                             


### 1. Prerequisites

**System Requirements:**
- Python 3.9+
- 4GB+ RAM (8GB recommended for better models)
- Microphone for voice input (optional)

**Install Ollama** (Required for LLM):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

**Start Ollama and pull a model:**
```bash
# Start Ollama service
ollama serve

# In another terminal, pull Mistral model
ollama pull mistral

# Alternative models:
# ollama pull llama2
# ollama pull codellama
```

### 2. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# For Linux users, install additional dependencies for pyttsx3
sudo apt-get update
sudo apt-get install espeak espeak-data libespeak-dev

# For Mac users with audio issues
brew install portaudio
```

### 3. First Run - Test Individual Components

**Test Speech-to-Text:**
```bash
python stt.py
# Provide a test audio file (input.wav)
```

**Test Text-to-Speech:**
```bash
python tts.py
# Generates test audio files
```

**Test MCP Tools:**
```bash
python mcp_tools.py
# Tests weather and Wikipedia APIs
```

**Test Orchestrator:**
```bash
python orchestrator.py
# Tests LLM + MCP integration
```

### 4. Run the Voice Agent

**Single File Mode:**
```bash
# Process one audio file
python voice_agent.py --input input.wav --output response.wav
```

**Interactive Mode:**
```bash
# Continuous conversation
python voice_agent.py --interactive
```

**With Custom Settings:**
```bash
# Use different models
python voice_agent.py --model llama2 --whisper small --interactive

# Adjust speech rate
python voice_agent.py --tts-rate 120 --interactive
```

## 📖 Detailed Setup

### Step 1: Setting Up Your RAG Chain

The voice agent integrates with your existing RAG pipeline. You need to modify `voice_agent.py` to load your RAG chain:

```python
# In voice_agent.py, update the load_rag_chain() function:

def load_rag_chain():
    """Load your existing RAG chain."""
    try:
        # Import your RAG chain
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from rag_chain import RAGChain  # Your class name
        
        # Initialize your RAG chain
        rag = RAGChain()
        # OR: rag = RAGChain.from_config("config.yaml")
        
        return rag
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {e}")
        return None
```

**Your RAG chain should support:**
- Method to query: `rag.invoke(query)` or `rag.run(query)`
- Return format: String or dict with "answer" key

### Step 2: Recording Audio Input

**Option 1: Use existing audio files**
```bash
# Any format: wav, mp3, m4a, etc.
python voice_agent.py --input my_question.wav
```

**Option 2: Record audio with Python**
```python
# record_audio.py
import sounddevice as sd
import soundfile as sf

# Record 5 seconds of audio
duration = 5  # seconds
fs = 16000  # Sample rate
print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
sf.write('input.wav', recording, fs)
print("Saved to input.wav")
```

**Option 3: Use system tools**
```bash
# macOS
rec input.wav

# Linux with arecord
arecord -d 5 -f cd -t wav input.wav

# Windows with SoX
sox -d input.wav
```

### Step 3: Understanding the Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        Voice Agent Flow                      │
└─────────────────────────────────────────────────────────────┘

1. USER SPEAKS → Audio File (input.wav)
                      ↓
2. SPEECH-TO-TEXT (Whisper)
   - Converts audio to text
   - Output: "What's the weather in London?"
                      ↓
3. ORCHESTRATOR (LLM Brain)
   ├─ Analyzes intent
   ├─ Decides: Need weather tool
   └─ Generates: [TOOL: get_weather] {"location": "London"}
                      ↓
4. MCP TOOL EXECUTION
   - Calls Open-Meteo API
   - Returns: {"temp": 12, "conditions": "Cloudy"}
                      ↓
5. LLM FINAL RESPONSE
   - Combines tool results
   - Output: "The weather in London is 12°C and cloudy."
                      ↓
6. TEXT-TO-SPEECH (pyttsx3)
   - Converts text to audio
   - Saves: response.wav
                      ↓
7. PLAYS AUDIO → User hears response
```

## 🔧 Configuration

### Model Selection

**Whisper Models** (STT):
| Model  | Size  | Speed     | Accuracy |
|--------|-------|-----------|----------|
| tiny   | 39MB  | Fastest   | Basic    |
| base   | 74MB  | Fast      | Good     |
| small  | 244MB | Medium    | Better   |
| medium | 769MB | Slow      | High     |
| large  | 1.5GB | Slowest   | Best     |

**Ollama Models** (LLM):
| Model      | Size  | Use Case            |
|------------|-------|---------------------|
| mistral    | 4.1GB | General purpose     |
| llama2     | 3.8GB | Conversational      |
| codellama  | 3.8GB | Code understanding  |
| phi        | 2.7GB | Lightweight         |

**Change models:**
```bash
python voice_agent.py --model phi --whisper tiny --interactive
```

### TTS Voice Selection

```python
from tts import TextToSpeech

tts = TextToSpeech()

# List available voices
voices = tts.list_voices()
for v in voices:
    print(f"{v['index']}: {v['name']}")

# Set voice by index
tts.set_voice(1)  # Usually female voice
tts.speak("Hello with a different voice!")
```

## 🛠️ Advanced Usage

### Adding Custom MCP Tools

Edit `mcp_tools.py` to add new tools:

```python
def get_stock_price(self, symbol: str) -> Dict[str, Any]:
    """Get current stock price."""
    # Your implementation
    return {"price": 150.25, "symbol": symbol}

def get_tool_definitions(self) -> List[Dict[str, Any]]:
    """Update to include new tool."""
    tools = [
        # ... existing tools ...
        {
            "name": "get_stock_price",
            "description": "Get current stock price for a symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., AAPL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    ]
    return tools
```

### Integrating with Your RAG Pipeline

Your RAG chain needs to be callable. Example integration:

```python
# In your rag_chain.py
class RAGChain:
    def invoke(self, query: str) -> str:
        """
        Query the RAG pipeline.
        
        Args:
            query: User's question
            
        Returns:
            Answer string
        """
        # Your existing RAG logic
        docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        answer = self.llm.generate(context, query)
        return answer
```

### Conversation Memory

The orchestrator maintains conversation history automatically:

```python
# Access conversation history
history = agent.orchestrator.conversation_history

# Clear history
agent.orchestrator.reset_conversation()

# Adjust memory length
agent.orchestrator.max_history = 10  # Remember last 10 exchanges
```

## 📊 Example Conversations

### Example 1: Weather + Wikipedia

**Input Audio:** "Tell me about London and what's the weather there?"

**Processing:**
1. STT: "Tell me about London and what's the weather there"
2. LLM decides: Need Wikipedia + Weather
3. Executes:
   - `[TOOL: search_wikipedia] {"query": "London"}`
   - `[TOOL: get_weather] {"location": "London"}`
4. Combines results
5. TTS: "London is the capital of England... The current weather is 12°C and cloudy."

### Example 2: RAG + Action

**Input Audio:** "What do our documents say about AI, and search Wikipedia for more info?"

**Processing:**
1. STT: "What do our documents say about AI..."
2. LLM decides: Need RAG + Wikipedia
3. Executes:
   - `[RAG_SEARCH: AI information]`
   - `[TOOL: search_wikipedia] {"query": "Artificial Intelligence"}`
4. Combines RAG results + Wikipedia
5. Generates comprehensive response

### Example 3: Multi-turn Conversation

```
User: "What's the weather in Paris?"
Agent: "The weather in Paris is 15°C and sunny."

User: "How about London?" (context maintained)
Agent: "In London, it's 12°C and cloudy."

User: "Which one is warmer?" (remembers both)
Agent: "Paris is warmer at 15°C compared to London's 12°C."
```

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to Ollama
```
**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify it's running
ollama list
```

### Whisper Model Download Issues
```
Error: Failed to load Whisper model
```
**Solution:**
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper
python stt.py
```

### TTS No Audio Output
```
Error: pyttsx3 initialization failed
```
**Solution:**
```bash
# Linux
sudo apt-get install espeak espeak-data libespeak-dev

# Mac
brew install portaudio

# Verify
python -c "import pyttsx3; pyttsx3.init()"
```

### RAG Chain Not Loading
```
Warning: RAG chain not loaded
```
**Solution:** Update `load_rag_chain()` in `voice_agent.py` to import your actual RAG chain:
```python
from src.rag_chain import YourRAGClass
rag = YourRAGClass()
```

### Audio Format Issues
```
Error: Audio file not found or invalid format
```
**Solution:**
```bash
# Convert to WAV
ffmpeg -i input.mp3 input.wav

# Check file
file input.wav
```

## 🧪 Testing

### Unit Tests

```bash
# Test STT
python -c "from stt import SpeechToText; stt = SpeechToText(); print('✓ STT OK')"

# Test TTS
python -c "from tts import TextToSpeech; tts = TextToSpeech(); print('✓ TTS OK')"

# Test MCP Tools
python -c "from mcp_tools import MCPTools; tools = MCPTools(); print('✓ MCP OK')"

# Test Orchestrator
python -c "from orchestrator import Orchestrator; orch = Orchestrator(); print('✓ Orchestrator OK')"
```

### Integration Test

```bash
# Create test audio
echo "Hello, what is the weather in Paris?" | \
  python -c "from tts import TextToSpeech; import sys; \
  tts = TextToSpeech(); \
  tts.speak(sys.stdin.read(), 'test_input.wav')"

# Process through voice agent
python voice_agent.py --input test_input.wav --output test_output.wav
```

## 📚 API Reference

### VoiceAgent Class

```python
agent = VoiceAgent(
    model_name="mistral",      # Ollama model
    whisper_model="base",      # Whisper size
    tts_rate=150,              # Speech rate (WPM)
    rag_chain=rag              # Your RAG chain
)

# Process single file
result = agent.process_voice_input("input.wav", "output.wav")
# Returns: {
#     "transcribed_text": "...",
#     "response_text": "...",
#     "audio_file": "output.wav"
# }

# Interactive mode
agent.interactive_mode()
```

### Orchestrator Class

```python
orchestrator = Orchestrator(
    model_name="mistral",
    rag_chain=rag,
    mcp_tools=tools,
    max_history=5
)

# Process query
response = orchestrator.process_query("What's the weather?")

# Reset conversation
orchestrator.reset_conversation()

# Access history
history = orchestrator.conversation_history
```

### MCPTools Class

```python
tools = MCPTools()

# Get tool definitions
tool_defs = tools.get_tool_definitions()

# Execute tool
result = tools.execute_tool("get_weather", {"location": "London"})

# Format result
formatted = tools.format_tool_result(result)
```

## 🎯 Performance Tips

1. **Use smaller models for faster response:**
   ```bash
   python voice_agent.py --model phi --whisper tiny --interactive
   ```

2. **Pre-load models:**
   ```bash
   # Pull models before running
   ollama pull mistral
   python -c "from stt import SpeechToText; SpeechToText()"
   ```

3. **Optimize RAG retrieval:**
   - Limit retrieved documents to top 3-5
   - Use smaller chunk sizes
   - Cache frequent queries

4. **Reduce latency:**
   - Use local Whisper (no API calls)
   - Keep Ollama model in memory
   - Process audio in chunks for real-time

## 🤝 Contributing

To extend the voice agent:

1. **Add new MCP tools** in `mcp_tools.py`
2. **Modify orchestration logic** in `orchestrator.py`
3. **Integrate additional RAG sources** in your RAG chain
4. **Add new TTS voices** in `tts.py`

## 📄 License

This project uses:
- OpenAI Whisper (MIT)
- Ollama (MIT)
- pyttsx3 (MPL-2.0)
- LangChain (MIT)

## 🙏 Credits

Built with:
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [Ollama](https://ollama.com) for local LLMs
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for TTS
- [LangChain](https://langchain.com) for RAG

---

**Need help?** Open an issue or check the troubleshooting section above.

**Ready to build?** Start with `python voice_agent.py --interactive`! 🚀




(venv) akash@Akashs-MacBook-Air langchain-rag-project % ollama serve
Couldn't find '/Users/akash/.ollama/id_ed25519'. Generating new private key.
Your new public key is: 

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL2bKdabUcjekpEZOggLd8fuJbFrPrGh6ZVhJzukOPG9