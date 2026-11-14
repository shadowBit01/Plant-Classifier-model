@AGENTS.md

Claude Code specifics:
- Never use plan mode: in autopilot nobody approves plans. Think, then edit.
- `codegraph_explore` (CodeGraph MCP) shows project-root paths: edit the same path under `.lanes/<LANE>`.
- Check context with `/context`; if a thread is past one task card, wrap up and stop instead of continuing.
