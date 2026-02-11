import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    import pyperclip
except ImportError as exc:
    raise SystemExit("Missing dependency: pyperclip. Install with `pip install pyperclip`.") from exc

try:
    import sounddevice as sd
except ImportError as exc:
    raise SystemExit("Missing dependency: sounddevice. Install with `pip install sounddevice`.") from exc

try:
    from pynput.keyboard import Controller, Key, Listener
except ImportError as exc:
    raise SystemExit("Missing dependency: pynput. Install with `pip install pynput`.") from exc

from qwen_asr import Qwen3ASRModel


@dataclass
class AppState:
    recording: bool = False
    last_toggle_ts: float = 0.0


class F8TranscriptionService:
    def __init__(
        self,
        model_name_or_path: str,
        model_device_map: str,
        sample_rate: int,
        channels: int,
        language: Optional[str],
        max_new_tokens: int,
        max_inference_batch_size: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.language = language
        self.state = AppState()
        self.state_lock = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None

        self.jobs: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self.keyboard = Controller()

        print("Loading model...")
        self.model = Qwen3ASRModel.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map=model_device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        print("Model ready.")

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _audio_callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        del frames, t
        if status:
            print(f"Audio status: {status}")
        self.frames.append(indata.copy())

    def _start_recording(self) -> None:
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=0,
        )
        self.stream.start()
        print("[F8] Recording started...")

    def _stop_recording(self) -> Optional[np.ndarray]:
        if self.stream is None:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None

        if not self.frames:
            print("No audio captured.")
            return None

        audio = np.concatenate(self.frames, axis=0)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        elif audio.ndim == 2:
            audio = audio[:, 0]

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        print(f"[F8] Recording stopped. Captured {audio.shape[0] / self.sample_rate:.2f}s audio.")
        return audio

    def _paste_text(self, text: str) -> None:
        if not text:
            print("Empty transcription. Nothing pasted.")
            return

        pyperclip.copy(text)
        time.sleep(0.05)
        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press("v")
            self.keyboard.release("v")
        print("Transcription pasted at cursor.")

    def _worker_loop(self) -> None:
        while True:
            audio = self.jobs.get()
            if audio is None:
                return
            try:
                print("Transcribing...")
                result = self.model.transcribe(
                    audio=(audio, self.sample_rate),
                    language=self.language,
                )
                text = result[0].text.strip() if result else ""
                detected_lang = result[0].language if result else "unknown"
                print(f"Detected language: {detected_lang}")
                print(f"Text: {text}")
                self._paste_text(text)
            except Exception as exc:  # noqa: BLE001
                print(f"Transcription failed: {exc}")

    def toggle_recording(self) -> None:
        now = time.time()
        with self.state_lock:
            if now - self.state.last_toggle_ts < 0.25:
                return
            self.state.last_toggle_ts = now

            if self.state.recording:
                self.state.recording = False
                audio = self._stop_recording()
                if audio is not None:
                    self.jobs.put(audio)
            else:
                self.state.recording = True
                self._start_recording()

    def stop(self) -> None:
        with self.state_lock:
            if self.state.recording:
                self.state.recording = False
                self._stop_recording()
        self.jobs.put(None)
        self.worker.join(timeout=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background F8 mic transcription service using Qwen3ASRModel.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="Model name or local path")
    parser.add_argument("--model-device-map", default="cpu", help="Transformers device_map value")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--language", default=None, help="Force language, default auto-detect")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = F8TranscriptionService(
        model_name_or_path=args.model,
        model_device_map=args.model_device_map,
        sample_rate=args.sample_rate,
        channels=args.channels,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.max_inference_batch_size,
    )

    print("Service is running.")
    print("Press F8 to start/stop recording.")
    print("Press ESC to quit.")

    running = True

    def on_press(key) -> bool:
        nonlocal running
        if key == Key.f8:
            service.toggle_recording()
        elif key == Key.esc:
            print("ESC pressed. Shutting down...")
            running = False
            return False
        return True

    listener = Listener(on_press=on_press)
    listener.start()

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Shutting down...")
    finally:
        listener.stop()
        service.stop()


if __name__ == "__main__":
    main()
