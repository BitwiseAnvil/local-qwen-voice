# Spoken replies

This project provides the local-qwen-voice MCP server. When its tools are available
and the user has not requested silence, use `speak` to read a brief natural-language
summary of the final answer. Keep code, file paths, and long lists in the written
answer. Check `speech_status` to confirm the job reaches playing or completed;
report a failed job rather than claiming speech played successfully.

Send the entire spoken summary in ONE `speak` call. Do not call `speak` separately
for each sentence or poll until every chunk completes. The server streams audio
while generating and manages continuous playback. `playing` means audio has
started; let it finish without repeatedly polling. `first_playback_ms` measures
time until the first buffer was submitted to Windows audio, including queue time.

Use `stop_speaking` immediately when the user says stop or quiet, or interrupts speech,
and stay silent for the rest of that reply; resume speaking on the next answer. If the
user asks for no speech at all, stay silent until they request it again. Design new voices
only when asked. Use the voice named in the server's instructions, which comes from
`config/voice.json`.

When changing when or how assistants speak, update BOTH the `INSTRUCTIONS` constant in
`local_voice/mcp_server.py` (Claude) and `docs/codex-AGENTS.md` (the Codex copy, which
users install as `%USERPROFILE%\.codex\AGENTS.md`). See "When it speaks" in README.md.
