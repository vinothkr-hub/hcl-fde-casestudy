# Solstice Docs Q&A Assistant

A small, dependency-free Python CLI that answers questions with excerpts from
the supplied Solstice documentation. It abstains when its retrieval evidence
does not meet the documented threshold. There is no LLM or external service.

## Requirements

- Python 3.10 or newer
- No third-party packages

## Run the demo

From the repository root, print answers for all five supplied questions:

```powershell
python assistant.py --examples
```

Ask one question:

```powershell
python assistant.py --question "How long before a password reset link expires?"
```

Or start an interactive session and type `exit` to quit:

```powershell
python assistant.py
```

Answers include the source document ID and title. Unsupported questions receive
an explicit “I don't have an answer” response without a source citation.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Design and limitations

See [SOLUTION_DESIGN.md](./SOLUTION_DESIGN.md) for the problem assumptions,
retrieval and abstention approach, trade-offs, and proposed next steps.

- Search uses lightweight lexical matching, not semantic embeddings; it may
  miss synonyms or unusual phrasing.
- The abstention threshold is a heuristic calibrated against the five supplied
  examples, not a production confidence score.
- Answers return one matching sentence verbatim; compound questions may need
  multiple sentences.
- The articles are loaded from the provided JSON file at startup. There is no
  document editing, web interface, authentication, or production monitoring.