# 🧭 Start Here

Welcome. This page helps you choose a pace and a path before you dive in.

## Learning Schedules

| Plan | Pace | Best for |
|---|---|---|
| **30-day sprint** | Tracks A + C only, skip deep dives | You need working-agent skills fast |
| **60-day standard** | Tracks A → B → C, light Track D | Steady learners, side-project pace |
| **90-day complete** | All 5 tracks, all mini-projects & capstones | You want the full, portfolio-ready outcome |

*(Detailed week-by-week breakdowns for each plan will live here as the repo grows — contributions welcome.)*

## Pick Your Path

| You are... | Start at |
|---|---|
| Completely new to AI/LLMs | [Track A → Part I](../01-foundations-of-generative-ai/README.md) |
| Know prompting, want to call APIs and build real things | [Part II](../02-ai-engineering-fundamentals/README.md) |
| Ready to build agents that take actions | [Track C → Part IX](../09-understanding-agents/README.md) |
| Already built one agent, want production/scale/security | [Track D → Part XV](../15-advanced-and-production-rag/README.md) |
| Just want certifications or interview prep | [Track E → Part XXII](../22-certifications-and-career-roadmap/README.md) |

## Common Beginner Mistakes

1. Jumping to "build an agent" before understanding basic prompting and API mechanics.
2. Treating every hallucination as a bug to prompt away, instead of a structural property of how LLMs work (see Part I.5 and Part III.4).
3. Picking a framework (LangGraph, CrewAI, etc.) before understanding what an agent *is* — the framework isn't the concept.
4. Ignoring token costs until the first surprise bill.
5. Skipping structured output / schema validation, then wondering why production breaks on edge cases.
6. Fine-tuning a model to fix a problem that better prompting or RAG would have solved more cheaply.
7. Building a multi-agent system when a single well-designed agent (or a simple workflow) would have been simpler and more reliable.
8. No evaluation strategy — shipping an agent with no way to measure if it's actually getting better or worse over time.
9. No guardrails — letting an agent take real-world actions (send email, spend money, delete files) without a human checkpoint early on.
10. Treating certifications as a substitute for a working portfolio project.

## Terminology Cheat Sheet

A one-page glossary lives in the [Appendix](../appendix/README.md) — refer back to it anytime a term is unfamiliar.

## "What Changed This Month"

This section will track significant shifts in the ecosystem (new frontier models, framework releases, protocol updates). Structured as a running changelog once the repo is live — a great first-contribution target for collaborators.

---

⬅️ [Back to main roadmap](../README.md)
