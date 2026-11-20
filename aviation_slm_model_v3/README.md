# Aviation SLM V3: Ollama and RAG Integration

## 1. Overview

This folder contains the Ollama and Retrieval-Augmented Generation (RAG) integration developed for Aviation SLM V3.

The integration allows users to interact with the aviation incident information system through the Ollama command-line interface.

It combines the Aviation SLM V3 model with a Python-based retrieval system that searches prepared aviation incident records.

### Main Components

- Aviation SLM V3 language model
- Ollama for local model interaction
- Python-based RAG service
- Ollama-compatible proxy
- Prepared aviation incident dataset

**Important:** The RAG system operates separately from the model weights. Its retrieval logic is implemented in Python and connected to Ollama through a proxy.

## 2. Folder Structure

```text
aviation_slm_model_v3/
│
├── aviation_rag_cli.py
├── aviation_rag_chat.py
├── rag_service.py
├── ollama_rag_proxy.py
├── Modelfile
├── Start_Aviation_RAG_Chat.bat
└── README.md
```

## 3. System Architecture

The integration follows this workflow:

1. The user enters a question through the Ollama CLI.
2. The Ollama-compatible proxy receives the request.
3. The proxy forwards the question to the Python RAG service.
4. The RAG service searches the prepared aviation incident records.
5. The system retrieves the relevant information.
6. The response is returned through the proxy to the Ollama CLI.

### Architecture Diagram

```text
              USER
                |
                v
            OLLAMA CLI
                |
                v
       OLLAMA-COMPATIBLE PROXY
                |
                v
         PYTHON RAG SERVICE
                |
                v
       AVIATION INCIDENT INDEX
                |
                v
       RECORD RETRIEVAL
                |
                v
        VERIFIED RESPONSE
                |
                v
            OLLAMA CLI
```

## 4. Model Configuration

A separate Ollama model configuration was created using the original Aviation SLM V3 model.

| Feature | Details |
|---|---|
| Original Model | `aviation-slm-v3:latest` |
| Integrated Model | `aviation-slm-v3-rag:latest` |
| Platform | Ollama |
| Temperature | 0 |
| Context Length | 2048 |
| Retrieval | Python-based RAG |
| Interface | Ollama CLI |

The original model remains unchanged.

## 5. Dataset Information

The RAG service uses a prepared JSON search index containing aviation incident records.

During testing, the service loaded the following information:

| Dataset Information | Count |
|---|---:|
| Total loaded records | 9,503 |
| Records containing date fields | 674 |
| Unique incident dates | 592 |

The dataset index is not included in this GitHub folder. Users must configure the location of their own prepared index.

### Dataset Configuration

The default expected location is:

```text
aviation_slm_model_v3/data/stage20_32_search_index.json
```

Alternatively, users can specify a custom dataset location through the `AVIATION_INDEX_PATH` environment variable.

For Windows PowerShell:

```powershell
$env:AVIATION_INDEX_PATH = "C:\path\to\stage20_32_search_index.json"
```

The dataset must follow the JSON structure expected by `aviation_rag_cli.py`.

## 6. Supported Information

The retrieval system supports questions related to:

- Aircraft registration
- Total fatalities
- Airline or operator
- Incident cause
- People on board
- Incident location
- Aircraft type
- Incident summary

Date-based questions must contain exactly one valid date in:

```text
DD/MM/YYYY
```

## 7. Requirements

The integration requires:

- Python
- Ollama
- Aviation SLM V3 Ollama model
- Prepared aviation incident JSON index
- FastAPI
- Uvicorn
- Requests
- Pydantic

Install the required Python packages:

```bash
pip install fastapi uvicorn requests pydantic
```

Ollama must also be installed and running locally.

## 8. Running the Integration

The system uses three terminal windows.

### Step 1: Start Ollama

Start the Ollama application.

The default Ollama server address is:

```text
http://localhost:11434
```

Make sure the required integrated model is available:

```bash
ollama list
```

### Step 2: Start the RAG Service

Navigate to this folder:

```text
aviation_slm_model_v3
```

Run:

```bash
python rag_service.py
```

The RAG service runs at:

```text
http://127.0.0.1:8004
```

### Step 3: Start the Ollama-Compatible Proxy

Open another terminal in the same folder.

Run:

```bash
python ollama_rag_proxy.py
```

The proxy runs at:

```text
http://127.0.0.1:11435
```

### Step 4: Start the Ollama CLI

Open a third terminal.

Set the Ollama host:

```cmd
set OLLAMA_HOST=127.0.0.1:11435
```

Start the integrated model:

```bash
ollama run aviation-slm-v3-rag:latest
```

The user can now enter aviation-related questions through the Ollama CLI.

Keep the RAG service and proxy running while using the chatbot.

## 9. API Endpoints

### RAG Service

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks RAG service status |
| POST | `/chat` | Processes aviation-related questions |

### Ollama-Compatible Proxy

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Proxy status |
| HEAD | `/` | Root endpoint availability |
| GET | `/health` | Checks proxy status |
| GET | `/api/version` | Returns version information |
| GET | `/api/tags` | Lists the integrated model |
| POST | `/api/show` | Returns model information |
| POST | `/api/chat` | Handles chat requests |
| POST | `/api/generate` | Handles generation requests |

The proxy implements the endpoints required for this project. It is not a complete replacement for the Ollama server API.

## 10. Testing and Validation

The integrated system was tested with different questions.

| Test Case | Question | Expected Result | Status |
|---|---|---|---|
| Aircraft Registration | What was the aircraft registration on 14/09/2008? | VP-BKO | Passed |
| Incident Summary | What happened on 14/09/2008? | Correct incident information | Passed |
| Unrelated Question | Who is the Prime Minister of India? | Rejects unrelated question | Passed |
| Unavailable Date | What happened on 15/09/2099? | Information unavailable | Passed |
| Invalid Date | What happened on 31/02/2020? | Invalid date message | Passed |

## 11. Example Responses

### Example 1: Aircraft Registration

Question:

```text
What was the aircraft registration on 14/09/2008?
```

Response:

```text
Answer: Aircraft registration on 14/09/2008: VP-BKO
```

### Example 2: Incident Summary

Question:

```text
What happened in the aviation incident on 14/09/2008?
```

Retrieved information:

```text
Operator: АЭРОФЛОТ НОРД
Aircraft: Boeing 737-500
Location: Perm, Perm Krai
Total fatalities: 88
Cause: Human factor
```

### Example 3: Unrelated Question

Question:

```text
Who is the Prime Minister of India?
```

Response:

```text
I can only answer aviation incident-related questions.
```

### Example 4: Unavailable Date

Question:

```text
What happened in the aviation incident on 15/09/2099?
```

Response:

```text
The information is not available in the provided aviation records.
```

## 12. Limitations

- The prepared aviation incident dataset must be available locally.
- The RAG service and proxy must remain active while using the integrated CLI.
- The proxy supports the specific endpoints required by this project.
- Responses depend on the information available in the prepared dataset.
- Questions outside the aviation incident domain are rejected.
- Date-based questions require the `DD/MM/YYYY` format.
- The integration is intended for educational and research purposes.

The system is not a production-ready aviation information service. Its responses should not be used for flight operations, aircraft maintenance, or aviation safety decisions.

## 13. Author

**Lokesh Addepalli**

**Project:** Aviation SLM V3