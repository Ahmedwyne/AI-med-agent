# 🧠 MedAgent Crew

Welcome to **MedAgent Crew** — a multi-agent AI system built using [crewAI](https://crewai.com), designed to empower intelligent agents to work collaboratively on complex medical research and data analysis tasks.

This project is developed and maintained by **me**, and it's tailored for healthcare research, clinical trial summarization, drug data extraction, and more — powered by live medical data sources like **PubMed**, **ClinicalTrials.gov**, and **RxNorm**.

---

## 🚀 Features

- 🤖 Multi-agent collaboration with **crewAI**
- 🏥 Accesses **PubMed**, **Clinical Trials**, and **RxNorm** APIs
- ⚙️ Powered by an **MCP server** for task orchestration
- 📊 Custom YAML-based configuration for agents and tasks
- 📄 Outputs research results to a markdown `report.md`
- 🔑 Secure `.env`-based API key management

---

## 🧰 Tech Stack

- `Python >=3.10 <3.13`
- [crewAI](https://crewai.com) – multi-agent framework
- [UV](https://docs.astral.sh/uv/) – fast package manager
- PubMed API – biomedical literature search
- ClinicalTrials.gov API – clinical studies metadata
- RxNorm API – drug and dosage normalization
- MCP Server – multi-agent command & processing backend
