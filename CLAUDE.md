@AGENTS.md

Claude Code specifics:
- Use plan mode for anything touching more than 2 files.
- Use `codegraph_explore` (CodeGraph MCP) for navigation before Read/Grep.
- Check context with `/context`; if a thread is past one task card, wrap up and stop instead of continuing.
