# foundry-rag

A reusable enterprise RAG framework for Azure AI Foundry.

## Supported data sources

| Source | Strategy |
|---|---|
| PDF / DOCX | Hierarchical section to paragraph to token window |
| Confluence | Page and heading boundary |
| Slack / Teams | Thread boundary, root message preserved |
| SQL databases | Row-group natural language template |
| Git repositories | AST function and class boundary |
| SAP exports | Row-group natural language template |

## Status

| Sprint | Scope | Status |
|---|---|---|
| 1 | Chunkers + interfaces + config | 67 tests passing |
| 2 | Azure embedder + AI Search index | In progress |
| 3 | Source connectors | Planned |
| 4 | Pipeline orchestration + deploy | Planned |

## Built by

whyaidata.com - Fractional Chief AI Officer services.
