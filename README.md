# Industrial Automation Project Suggester Chatbot

A conversational assistant that helps students, hobbyists, and early-career engineers pick a suitable **industrial automation / mechatronics project** based on their skill level, available hardware, and interests — then helps scope it into an actionable plan.

## Status

This repository is currently a lightweight front door to the project: the working prototype, prompt materials, and supporting documents are hosted in the linked Google Drive folder below while the codebase is being migrated/cleaned up for a public release here.

**Project materials:** https://drive.google.com/drive/folders/1N4pYgPCTai7WTsT19mZgZ4YAcLNXoarw?usp=drive_link

## Concept

Picking a first (or next) automation/mechatronics project is harder than it sounds — the right choice depends on what parts you already have (PLC, Arduino/ESP32, sensors, motors), how much time you can commit, and what you actually want to learn. This chatbot is designed to:

- Ask a short set of questions about skill level, available hardware/budget, and interests (e.g. robotics, process control, IoT monitoring).
- Suggest a shortlist of concrete project ideas, ranked by fit.
- Break the chosen idea down into a rough bill of materials and a milestone plan.

## Roadmap

- [ ] Migrate the chatbot code/notebooks from Drive into this repository
- [ ] Add a `requirements.txt` / environment setup
- [ ] Add example conversations and sample project suggestions
- [ ] Add a simple web or CLI front end

## Contributing

Suggestions and project-idea contributions are welcome — open an issue describing the industrial automation project idea you'd like to see supported, or a pull request once code lands in this repo.

## License

MIT — see [LICENSE](LICENSE).
