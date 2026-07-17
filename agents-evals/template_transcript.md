# Transcript Template

Save full conversations as `<prompt-id>--<model>--<skills>--<run>.md`

## Naming convention

```
<prompt-id>--<model>--(with-skills|bare)--<run>.md
```

Examples:
- `causality-01--deepseek-v4--with-skills--run1.md`
- `navier-stokes-sensor--claude-opus-4--bare--run1.md`
- `l-shaped-region--fable--with-skills--run3.md`

## What to capture

1. **Your full prompt** (the text from `prompts/*.yaml`)
2. **The model's full response** — all text, code blocks, and tool calls
3. **Do not edit or truncate** — the checker parses the raw output