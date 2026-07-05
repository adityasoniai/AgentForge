# 📎 Appendix

## A.1 Glossary of Terms

| Term | Plain-English Definition |
|---|---|
| **LLM** | Large Language Model — predicts text one token at a time |
| **Token** | A chunk of text (roughly ¾ of a word) that models process |
| **Context Window** | The maximum amount of text (in tokens) a model can "see" at once |
| **Hallucination** | When a model states something false with confidence, because it's predicting plausible text, not looking up facts |
| **RAG** | Retrieval-Augmented Generation — giving a model relevant real data before it answers |
| **Embedding** | A list of numbers representing the "meaning" of text, used for search |
| **Agent** | A system where a model reasons, calls tools, and acts across multiple steps toward a goal |
| **MCP** | Model Context Protocol — an open standard for connecting models to tools and data |
| **A2A** | Agent-to-Agent protocol — an open standard for agents built on different frameworks to communicate |
| **Fine-tuning** | Further training a pretrained model on your own data to specialize it |
| **LoRA / QLoRA** | Efficient fine-tuning methods that update a small number of parameters instead of the whole model |
| **Guardrails** | Constraints placed on an agent to prevent unsafe or unwanted actions |

*(This glossary will keep growing — add a term any time a section introduces new jargon.)*

## A.2 Master Free vs. Paid Comparison Table

> Centralized here instead of repeated per-section — sections link back to relevant rows.

| Category | Free / Open | Paid / Managed |
|---|---|---|
| Vector DBs | Chroma, Qdrant (self-hosted), pgvector | Pinecone, Weaviate Cloud |
| Agent Frameworks | LangGraph, CrewAI, Google ADK, Claude Agent SDK, OpenAI Agents SDK (all open source) | LangGraph Platform, CrewAI Enterprise (optional add-ons) |
| Local Model Serving | Ollama, LM Studio, vLLM | Managed inference (Bedrock, Vertex AI, Azure OpenAI) |
| Observability | Phoenix (open source), OpenTelemetry | LangSmith, Helicone, Arize (paid tiers) |
| Workflow Platforms | n8n (self-hosted) | Zapier AI, Dify Cloud, Copilot Studio |
| Certifications | LangChain Academy, Anthropic Academy, Google free learning paths | Google Cloud GenAI Engineer exam, AWS certs, Azure AI-901 |

*(This table is a living document — as we build each section, its specific comparisons will be added here.)*

## A.3 Curated Research Papers

- Wei et al., *Chain-of-Thought Prompting* (2022)
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2023)
- Yao et al., *Tree of Thoughts* (2023)
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)

*(Expanding as each Part is written — advanced readers, contribute your favorites!)*

## A.4 Visual Architecture Diagrams Index

A running index of every diagram in the repo will live here once more sections are complete, so you can jump straight to visual explanations.

## A.5 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

⬅️ [Back to main roadmap](../README.md)
