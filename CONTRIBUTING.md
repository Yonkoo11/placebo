# Contributing

- Run `python3 -m pytest -q tests` before a pull request. A change that alters a trial number must say why in the pull request.
- The verdict rule in `placebo/engine.py` (`verdict`) is fixed before any run. Changing it means re-running every published trial and saying so.
- Never add data from before 2026-07-01 to the agent window without also changing the agent model to one released before that data.
- Keys come from the environment (`GEMINI_API_KEY`). Never commit one.
