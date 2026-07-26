# Security

- Run with `REACTS_REQUIRE_API_KEY=true` and a strong `REACTS_API_KEY` outside an isolated local environment.
- Keep source artifacts, model artifacts, and registry volumes outside the container image.
- Treat patent data and uploaded batch files as untrusted input.
- Do not expose the training or index-build endpoints publicly without authentication, rate limits, and a durable isolated worker.
- Rotate API keys and back up the registry before model promotion or dataset replacement.
- Model outputs are scientific decision support, not validated laboratory instructions.
