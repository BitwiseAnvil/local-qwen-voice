"""Host commands for readiness checks and a real GPU synthesis smoke test."""

import argparse
from array import array
from contextlib import closing
import io
import json
import math
from pathlib import Path
import sys
import statistics
import threading
import time
import uuid
import wave

from .client import EngineClient
from .common import wav_info
from .playback import WindowsPlayer
from .settings import backend_url, default_voice
from .streaming import pcm_to_wav

SAMPLE_TEXT = "Hello. This is your local voice speaking as the audio is generated. You should hear the first words while the rest of the sentence is still being prepared."


def wait_ready(client, timeout):
    deadline = time.monotonic() + timeout
    last = "No response yet"
    while time.monotonic() < deadline:
        try:
            state = client.health()
        except RuntimeError as exc:
            last = str(exc)
        else:
            if state.get("status") == "ready":
                if state.get("streaming_protocol") != 1:
                    raise RuntimeError("An old non-streaming Docker image is running. Rebuild with scripts/start.ps1.")
                return state
            if state.get("status") == "failed":
                raise RuntimeError(state.get("error") or "Engine initialization failed")
            last = state.get("status", "loading")
        time.sleep(2)
    raise RuntimeError(f"TTS did not become ready in {timeout}s: {last}")


def verify_sample(audio, metrics):
    info = wav_info(audio)
    if metrics.get("device") != "cuda:0":
        raise RuntimeError("Synthesis was not reported on the CUDA GPU")
    if metrics.get("cuda_parameters_verified", 0) <= 0 or metrics.get("peak_cuda_memory_mb", 0) <= 0:
        raise RuntimeError("Synthesis did not include actual CUDA tensor and memory evidence")
    with wave.open(io.BytesIO(audio), "rb") as stream:
        samples = array("h", stream.readframes(stream.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(float(sample) ** 2 for sample in samples) / len(samples)) / 32768
    if rms < 0.0001:
        raise RuntimeError("Generated sample is silent")
    return dict(info, rms=round(rms, 6), **metrics)


def stream_sample(client, text, voice, player=None):
    """Measure the same framed HTTP/Windows audio path used by the MCP bridge."""
    identifier = uuid.uuid4().hex
    cancel = threading.Event()
    started = time.perf_counter()
    parts = []
    metrics = None
    first_received = None
    first_playback = None
    try:
        with closing(client.stream_synthesize(text, voice, identifier, cancel)) as frames:
            for kind, value in frames:
                if kind == "H" and player:
                    player.open_stream(value["sample_rate"])
                elif kind == "A":
                    if first_received is None:
                        first_received = (time.perf_counter() - started) * 1000
                    parts.append(value)
                    if player:
                        player.wait_capacity(cancel)
                        player.write(value)
                        if first_playback is None:
                            first_playback = (time.perf_counter() - started) * 1000
                elif kind == "M":
                    metrics = value
        if metrics is None or not parts:
            raise RuntimeError("Speech stream ended without completion metrics")
        received_seconds = time.perf_counter() - started
        if player:
            player.finish(cancel)
        result = dict(metrics, first_audio_received_ms=round(first_received, 1),
                      receive_seconds=round(received_seconds, 3),
                      delivery_real_time_factor=round(received_seconds / metrics["audio_seconds"], 3))
        if player:
            result["first_playback_ms"] = round(first_playback, 1)
            result["playback_underruns"] = player.underruns
        return pcm_to_wav(b"".join(parts)), result
    except BaseException:
        cancel.set()
        if player:
            player.stop()
        client.close_stream(identifier)
        try:
            client.cancel(identifier)
        except Exception:
            pass
        raise
    finally:
        if player:
            player.stop()


def verify_interactive(metrics, max_first_audio_ms=1000):
    if (metrics.get("backend") != "faster-qwen3-tts" or not metrics.get("cuda_graphs")
            or not metrics.get("streaming") or metrics.get("audio_chunks", 0) < 2):
        raise RuntimeError("The sample did not exercise accelerated streaming with captured CUDA graphs")
    first = metrics.get("first_playback_ms", metrics.get("first_audio_received_ms", float("inf")))
    if first > max_first_audio_ms:
        raise RuntimeError(f"First audio took {first:.0f} ms; target is {max_first_audio_ms:.0f} ms")
    if metrics.get("real_time_factor", float("inf")) >= 1:
        raise RuntimeError("Synthesis is still slower than playback; inspect the benchmark report")
    if metrics.get("delivery_real_time_factor", 0) >= 1:
        raise RuntimeError("Audio delivery is still slower than playback; inspect the benchmark report")
    if metrics.get("playback_underruns", 0):
        raise RuntimeError("The Windows playback buffer ran out of audio during the sample")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("health", "wait", "smoke", "benchmark"))
    parser.add_argument("--url", default=backend_url(project))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--text", default=SAMPLE_TEXT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-first-audio-ms", type=float, default=1000)
    parser.add_argument("--require-interactive", action="store_true")
    args = parser.parse_args()
    client = EngineClient(args.url)
    try:
        if args.command == "health":
            result = client.health()
        elif args.command == "wait":
            print("Waiting for the GPU models and designed voice...", flush=True)
            result = wait_ready(client, args.timeout)
        else:
            state = client.health()
            if state.get("status") != "ready":
                raise RuntimeError(state.get("error") or "Engine is still loading")
            voice = default_voice(project)
            runs = args.runs if args.command == "benchmark" else 1
            if not 1 <= runs <= 10:
                raise ValueError("--runs must be between 1 and 10")
            reports = []
            for index in range(runs):
                # Only the final run is audible; earlier runs measure delivery
                # without playback backpressure. Model warmup happened at startup.
                player = WindowsPlayer() if args.play and index == runs - 1 else None
                audio, metrics = stream_sample(client, args.text, voice, player)
                reports.append(verify_sample(audio, metrics))
                print(f"Sample {index + 1}/{runs}: first PCM {metrics['first_audio_received_ms']} ms, RTF {metrics['real_time_factor']} (lower is faster)", file=sys.stderr)
            result = reports[-1]
            output = project / ".runtime" / "voice-sample.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(audio)
            result["sample_file"] = str(output)
            if args.command == "benchmark":
                result = {"runs": reports, "sample_file": str(output),
                          "median_first_audio_ms": statistics.median(r["first_audio_received_ms"] for r in reports),
                          "median_real_time_factor": statistics.median(r["real_time_factor"] for r in reports)}
                report_path = project / ".runtime" / "streaming-benchmark.json"
                report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                result["report_file"] = str(report_path)
            # Print measured evidence even if the requested performance check fails.
            print(json.dumps(result, indent=2))
            if args.require_interactive:
                for report in reports:
                    verify_interactive(report, args.max_first_audio_ms)
            return 0
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
