# Spoken replies

<!-- Copy this into %USERPROFILE%\.codex\AGENTS.md (merge with any existing content).
     If you changed the default voice in config/voice.json, update the voice name below. -->

The `local-qwen-voice` MCP server provides local speech on this computer. If its tools
are not already loaded, find them with tool search (`speak`, `speech_status`,
`stop_speaking`).

When to speak:
1. After every final answer, give a 1-3 sentence spoken summary of the key outcome.
2. When you are blocked waiting for the user's answer, approval, or decision, briefly say what you need.
3. When a long-running task such as a build, test run, or install finishes or fails, announce the result.

Speak plain natural language only. Keep code, file paths, commands, tables, and long
lists in the written reply. Always pass voice `audition_smooth_rounded` unless the user
asks for a different voice.

Send each spoken message in ONE `speak` call; the server streams it. Do not split it into
sentence-by-sentence calls. After queueing, check `speech_status` once; `playing` means
audio has started, so do not keep polling. If the job failed, say so instead of claiming
speech played.

Silence: when the user says stop or quiet, or interrupts, call `stop_speaking` immediately
and stay silent for the rest of that reply; resume speaking on the next answer. If the user
asks for no speech at all, stay silent until they ask for it again.
