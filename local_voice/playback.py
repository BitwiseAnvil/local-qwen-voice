"""Continuous Windows PCM playback using WinMM; no host pip packages required."""

from collections import deque
import ctypes as ct
import io
import os
import threading
import wave

from .common import Cancelled, wav_info
from .streaming import MAX_FRAME


class WaveFormat(ct.Structure):
    _fields_ = [("tag", ct.c_uint16), ("channels", ct.c_uint16),
                ("rate", ct.c_uint32), ("bytes_per_second", ct.c_uint32),
                ("block_align", ct.c_uint16), ("bits", ct.c_uint16), ("extra", ct.c_uint16)]


class WaveHeader(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("length", ct.c_uint32),
                ("recorded", ct.c_uint32), ("user", ct.c_size_t),
                ("flags", ct.c_uint32), ("loops", ct.c_uint32),
                ("next", ct.c_void_p), ("reserved", ct.c_size_t)]


class WinMM:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Run the MCP bridge on Windows for speaker playback")
        self.dll = ct.WinDLL("winmm")
        handle = ct.c_void_p
        self.dll.waveOutOpen.argtypes = [ct.POINTER(handle), ct.c_uint32,
                                        ct.POINTER(WaveFormat), ct.c_size_t, ct.c_size_t, ct.c_uint32]
        for name in ("waveOutPrepareHeader", "waveOutUnprepareHeader", "waveOutWrite"):
            getattr(self.dll, name).argtypes = [handle, ct.POINTER(WaveHeader), ct.c_uint32]
        for name in ("waveOutReset", "waveOutClose"):
            getattr(self.dll, name).argtypes = [handle]
        for name in ("waveOutOpen", "waveOutPrepareHeader", "waveOutUnprepareHeader",
                     "waveOutWrite", "waveOutReset", "waveOutClose"):
            getattr(self.dll, name).restype = ct.c_uint32
        self.dll.waveOutGetErrorTextW.argtypes = [ct.c_uint32, ct.c_wchar_p, ct.c_uint32]

    def call(self, name, *args):
        result = getattr(self.dll, name)(*args)
        if result:
            message = ct.create_unicode_buffer(256)
            self.dll.waveOutGetErrorTextW(result, message, len(message))
            raise RuntimeError(f"{name}: {message.value or result}")


class WindowsPlayer:
    def __init__(self, directory=None, *, api=None, max_buffer_seconds=3):
        self.api = api
        self.handle = None
        self.buffers = deque()
        self.lock = threading.RLock()
        self.rate = 24000
        self.max_buffer_seconds = max_buffer_seconds
        self.underruns = 0
        self.frames_written = 0

    def open_stream(self, sample_rate=24000):
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
            raise ValueError("Invalid playback sample rate")
        with self.lock:
            self.stop()
            if self.api is None:
                self.api = WinMM()
            handle = ct.c_void_p()
            audio_format = WaveFormat(1, 1, sample_rate, sample_rate * 2, 2, 16, 0)
            self.api.call("waveOutOpen", ct.byref(handle), 0xFFFFFFFF,
                          ct.byref(audio_format), 0, 0, 0)
            self.handle = handle
            self.rate = sample_rate
            self.underruns = 0
            self.frames_written = 0

    def _reap(self):
        while self.buffers and self.buffers[0][0].flags & 1:  # WHDR_DONE
            header, data = self.buffers[0]
            self.api.call("waveOutUnprepareHeader", self.handle, ct.byref(header), ct.sizeof(header))
            self.buffers.popleft()  # Retain header AND PCM allocation until unprepared.

    def write(self, pcm):
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > MAX_FRAME:
            raise ValueError("Expected a bounded frame of mono PCM16 audio")
        with self.lock:
            if self.handle is None:
                raise Cancelled("Audio playback is closed")
            self._reap()
            if not self.buffers and self.frames_written:
                self.underruns += 1
            data = ct.create_string_buffer(pcm, len(pcm))
            header = WaveHeader()
            header.data = ct.addressof(data)
            header.length = len(pcm)
            self.api.call("waveOutPrepareHeader", self.handle, ct.byref(header), ct.sizeof(header))
            self.buffers.append((header, data))
            try:
                self.api.call("waveOutWrite", self.handle, ct.byref(header), ct.sizeof(header))
            except BaseException:
                self.api.call("waveOutUnprepareHeader", self.handle, ct.byref(header), ct.sizeof(header))
                self.buffers.pop()
                raise
            self.frames_written += 1

    def _wait(self, cancel, *, drain):
        while True:
            if cancel.is_set():
                raise Cancelled("Speech playback cancelled")
            with self.lock:
                if self.handle is None:
                    raise Cancelled("Audio playback is closed")
                self._reap()
                pending = sum(header.length for header, _ in self.buffers)
                ready = not pending if drain else pending < self.rate * 2 * self.max_buffer_seconds
                if ready:
                    return
            # Never hold the player/controller lock while waiting for the device.
            cancel.wait(0.005)

    def wait_capacity(self, cancel):
        self._wait(cancel, drain=False)

    def finish(self, cancel):
        self._wait(cancel, drain=True)

    def stop(self):
        with self.lock:
            if self.handle is None:
                return
            self.api.call("waveOutReset", self.handle)
            while self.buffers:
                header, data = self.buffers[0]
                # Keep the allocation if the driver refuses to release a header.
                self.api.call("waveOutUnprepareHeader", self.handle, ct.byref(header), ct.sizeof(header))
                self.buffers.popleft()
            self.api.call("waveOutClose", self.handle)
            self.handle = None

    def start(self, audio):
        """Compatibility for callers playing an already generated WAV."""
        info = wav_info(audio)
        self.open_stream(info["sample_rate"])
        with wave.open(io.BytesIO(audio), "rb") as stream:
            while pcm := stream.readframes(24000):
                self.write(pcm)
        return info["duration_seconds"]
