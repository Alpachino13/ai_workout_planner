# Standalone Workout Planner

This is separate from the course notebooks. It is a small command-line agent
powered by Gemini through `langchain-google-genai`.

## Configure

Put your Gemini key in the repository's existing `.env` file under:

```env
GOOGLE_API_KEY=your_key_here
```

The agent does not create, modify, or print API keys.

## Run

From the repository root:

```powershell
uv run python -m workout_planner
```

Type `exit` to stop.

