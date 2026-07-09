# Industrial Automation Project Suggester (Chatbot)

A chatbot concept that recommends industrial automation and mechatronics project ideas tailored to a user's skill level and available hardware.

<p>
  <img alt="Status" src="https://img.shields.io/badge/status-concept%2Fprototype-orange">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

## Concept

Students and hobbyists in mechatronics/automation often have hardware on hand (an Arduino, a few sensors, a motor driver) but aren't sure what to build with it. This project's idea is a conversational assistant that asks a few questions — skill level, available components, time budget — and suggests concrete, scoped project ideas, with a rough difficulty and parts list for each.

## Status

This repository currently captures the project concept rather than a deployed application; the working prototype and supporting materials are being migrated in from local development. Planned scope:

- A question flow to capture skill level, available hardware, and interests
- A recommendation engine (rule-based or LLM-backed) that maps that input to a curated set of project ideas
- A simple chat interface (CLI or web) for interacting with the suggester

## Related Projects

For working examples of a similar chatbot pattern in this profile, see [Rule-Based-Chat-Bot](https://github.com/Abdelrahmanemad0/Rule-Based-Chat-Bot) (TF-IDF + Gradio UI).

## License

MIT — see [LICENSE](LICENSE).
