# Ollama Proxy

A debugging proxy server for the Ollama API that provides tool injection capabilities and advanced request/response logging.

## Features

- **Tool Injection**: Automatically inject tools into chat requests to enable function calling
- **Qwen Model Support**: Specialized stream parser for intercepting XML-based tool calls from Qwen models
- **Comprehensive Logging**: Detailed request/response logging with configurable log levels
- **Title Prompt Replacement**: Automatic title generation for long conversations
- **Request Filtering**: Exclude specific patterns from tool injection
- **Thread-Safe**: Multi-threaded HTTP server for concurrent requests

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Install dependencies
pip install -e .

# Or using pyproject.toml directly
pip install python-dotenv
```

### Docker

```bash
docker build -t ollama-proxy .
docker run -p 11435:11435 -e OLLAMA_HOST=host.docker.internal -e OLLAMA_PORT=11434 ollama-proxy
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_HOST` | `127.0.0.1` | Host to bind the proxy server |
| `LISTEN_PORT` | `11435` | Port for the proxy server |
| `OLLAMA_HOST` | `127.0.0.1` | Backend Ollama API host |
| `OLLAMA_PORT` | `11434` | Backend Ollama API port |
| `LOG_FILE` | `ollama_proxy.log` | Log file path |
| `LOG_LEVEL` | `FULL` | Logging level (OFF, CRITICAL, BRIEF, NORMAL, FULL) |
| `ENABLE_TOOL_INJECTION` | `True` | Enable/disable tool injection |
| `TOOLS_FILE` | `ollama_tools.json` | Path to tools JSON file |
| `TOOL_INJECTION_EXCLUDE_PATTERN` | - | Regex pattern to exclude from tool injection |

## Usage

### Basic Usage

```bash
# Start the proxy
ollama-proxy

# Or with environment variables
export OLLAMA_HOST=host.docker.internal
export OLLAMA_PORT=11434
ollama-proxy
```

### Docker Usage

```bash
docker run -p 11435:11435 \
  -e OLLAMA_HOST=host.docker.internal \
  -e OLLAMA_PORT=11434 \
  ollama-proxy
```

## How It Works

### Request Flow

1. Client sends request to proxy on port 11435
2. Proxy intercepts `/api/chat` requests
3. Tool injection is applied if enabled and not excluded
4. Title prompt replacement for long conversations (if applicable)
5. Forwarded to backend Ollama API
6. Response stream is parsed and logged
7. Clean response sent back to client

### Stream Parsing

The proxy includes a custom `QwenStreamParser` that:

- Parses streaming responses chunk by chunk
- Intercepts XML-based tool calls from Qwen models
- Converts them to native Ollama tool call format
- Filters out hallucinated tools not in the allowed list
- Maintains stream integrity for passthrough requests

### Tool Injection

Tools are injected into chat requests as follows:

```json
{
  "messages": [...],
  "tools": [
    {
      "function": {
        "name": "run_terminal_command",
        "description": "Run a terminal command",
        "parameters": {...}
      }
    }
  ]
}
```

## Logging

Logs are written to the configured `LOG_FILE` and include:

- Request details (path, model, stream status)
- Tool information (names, descriptions, parameters)
- Message content and structure
- Response chunks and tool calls
- Errors and warnings

Log levels control verbosity from OFF to FULL.

## Project Structure

```
ollama-proxy/
├── src/
│   └── ollama_proxy/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py          # Main application logic
├── Dockerfile
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

MIT License - See [LICENSE](LICENSE) file for details.
