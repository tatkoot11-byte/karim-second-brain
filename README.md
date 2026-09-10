# Karim's Second-Brain Agent

A local AI second-brain assistant built with **LangGraph, mem0, Gemini, Chainlit, and DeepEval**.

The project demonstrates stateful agent orchestration, tool usage, local memory, consent-gated memory writes, recovery/fallback behavior, and automated evaluation.

## Features

* LangGraph stateful agent workflow
* Typed agent state with:

  * messages
  * intermediate tool results
  * step counter
  * tool-call counter
* ReAct-style reasoning loop
* Planner/Executor routing pattern
* Controlled graph cycles
* Pydantic-based tool input validation
* Web search, calculator, and note-saving tools
* Per-run safety limits:

  * `MAX_TOOL_CALLS = 4`
  * `MAX_STEPS = 6`
* Graceful recovery and fallback on invalid tool input or repeated failures
* Local mem0 memory with Gemini embeddings
* Memory categories:

  * `profile`
  * `task`
  * `episodic`
* Explicit consent required before memory storage
* `/forget` command for deleting all memories
* Chainlit user interface
* Visible tool-call and memory-retrieval steps
* DeepEval evaluation suite
* Answer Relevancy metric with threshold `0.7`
* Custom GEval `ConsentRespect`
* Tool-call correctness checking
* JSONL evaluation dataset
* Local-only and zero-cost design

## Architecture

```text
User
  |
  v
Chainlit UI
  |
  +----> Memory Retrieval
  |          |
  |          v
  |        mem0
  |
  v
LangGraph
  |
  +--> Planner
  |      |
  |      v
  |    Tools
  |      |
  |      v
  |    ReAct
  |      |
  |      +------> Planner (controlled cycle)
  |      |
  |      v
  |    Respond
  |
  +--> Fallback
  |
  v
Final Response
```

## Project Structure

```text
karim-second-brain/
│
├── agent/
│   ├── graph.py
│   ├── memory.py
│   └── tools.py
│
├── data/
│   └── eval_cases.jsonl
│
├── docs/
│   ├── chainlit_steps.png
│   ├── demo_consent_no.png
│   └── demo_consent_yes.png
│
├── tests/
│   └── test_agent.py
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key
```

The API key is kept outside the repository and `.env` is excluded through `.gitignore`.

## Running the Agent

Run a direct LangGraph test:

```bash
python -c "from agent.graph import run_agent; result=run_agent('What is 15 + 7?'); print(result['messages'][-1].content)"
```

Expected behavior:

```text
Transition: planner
Planner selected: calculator
Transition: tools
Tool call: calculator
Transition: react
Transition: respond
```

The agent also prints step counts, tool-call counts, and recovery attempts.

## Safety and Recovery

The agent enforces:

```text
MAX_TOOL_CALLS = 4
MAX_STEPS = 6
```

Invalid tool input is validated using Pydantic.

If a tool fails, the agent enters a recovery path and may cycle back through the planner. Repeated failures eventually trigger a safe fallback instead of continuing indefinitely.

Example fallback behavior:

```text
Recovery: Calculator failed gracefully
Transition: react
Transition: planner
Transition: tools
Recovery: Calculator failed gracefully
Transition: fallback
Fallback: stopped safely without further tool calls.
```

## Memory

The project uses a self-hosted local mem0 memory configured with Gemini.

The fixed user ID is:

```text
karim
```

Memory categories:

* `profile` — stable user preferences
* `task` — project and task-related information
* `episodic` — events and previous interactions

### Consent Gate

Memory writes are never performed without explicit user consent.

The application displays:

```text
Would you like me to store this request in Karim's memory?

[Yes, store it] [No, don't store it]
```

Only an affirmative response allows `memory.add()` to run.

### Forget Command

The Chainlit application supports:

```text
/forget
```

This deletes all memories associated with the user and reports the number deleted without exposing memory contents.

## Chainlit UI

Start the application with:

```bash
chainlit run app.py
```

The application runs locally at:

```text
http://localhost:8000
```

The UI shows:

* memory retrieval
* tool calls
* recovery steps
* final responses
* consent actions

## DeepEval Evaluation

The evaluation dataset is stored in:

```text
data/eval_cases.jsonl
```

The evaluation includes multiple test cases covering:

* calculator tool use
* web search
* note handling
* invalid input recovery
* tool-call correctness
* memory-related behavior

Metrics:

### Answer Relevancy

Threshold:

```text
0.7
```

### ConsentRespect

A custom GEval criterion verifies that the agent never claims that a memory was stored unless the user explicitly consented.

### ToolCorrectness

The expected tool and the tool actually called are compared for each applicable test case.

DeepEval supports evaluating LangGraph workflows and tracing graph, model, and tool execution.

Run the evaluation with:

```bash
python -m tests.test_agent
```

The evaluation produces:

```text
eval_report.json
```

## Demo Scenario

The required end-to-end demo uses:

```text
Remember my preferred tone is concise. Summarize the tool outputs, then remind me of the preference in the final answer — but only if I consented to storing it.
```

The demo demonstrates:

1. Agent execution
2. Tool activity
3. Memory retrieval
4. Explicit consent request
5. Memory storage only after `Yes`
6. Final response respecting the stored preference

### Demo Evidence

Demo evidence is stored under:

```text
docs/
├── chainlit_steps.png
├── demo_consent_no.png
└── demo_consent_yes.png
```

* `chainlit_steps.png` — Chainlit workflow and visible agent steps
* `demo_consent_no.png` — demonstration of the no-consent path
* `demo_consent_yes.png` — demonstration of explicit consent and successful memory storage

## Requirements

* Python 3.13
* Gemini API key
* Internet connection for Gemini and web-search functionality
* Windows, macOS, or Linux
* No paid services required

## Privacy

This project is designed for local development.

* API credentials are stored in `.env`
* `.env` is excluded from Git
* Memory uses the local mem0/Qdrant setup
* No deployment is required
* No streaming is required

## Acceptance Checklist

* [x] LangGraph typed state
* [x] ReAct pattern
* [x] Planner/Executor pattern
* [x] Controlled graph cycle
* [x] Pydantic validation
* [x] Tool-call limits
* [x] Step limits
* [x] Recovery and fallback
* [x] mem0 local memory
* [x] Profile/task/episodic memory categories
* [x] Explicit consent gate
* [x] `/forget`
* [x] Chainlit UI
* [x] Visible tool and memory steps
* [x] DeepEval test suite
* [x] Answer Relevancy threshold
* [x] Custom ConsentRespect GEval
* [x] Tool-call correctness check
* [ ] Final DeepEval report
* [x] Demo screenshots/recording
* [ ] GitHub repository
