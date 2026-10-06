# Workout Planner (Streamlit + LangChain + Gemini)

```bash
uv run python -m workout_planner      # or: streamlit run workout_planner/app.py
```

## Keys
Locally: `.env` (see `.env.example`). On Streamlit Cloud: *App settings → Secrets*
(see `.streamlit/secrets.toml.example`). `.env` is not deployed.

## LangSmith tracing
Set `LANGSMITH_API_KEY`; tracing turns on automatically (`LANGSMITH_TRACING=false` forces it off).
The sidebar **Monitoring** badge does a real authentication call and tells you what is wrong.

| Symptom | Cause |
|---|---|
| "Tracing off" | no key in Secrets/.env, or placeholder value |
| 401 | invalid/revoked key |
| 403 | wrong region (EU is auto-detected) or org key without `LANGSMITH_WORKSPACE_ID` |
| Traces in another project | `LANGSMITH_PROJECT` set elsewhere (default: `workout-planner`) |

One user action = one trace (`workout_planner_turn` / `_approve` / `_decline` / `_feedback`),
grouped per conversation by `thread_id`. Attached photos/audio are truncated in traces.
