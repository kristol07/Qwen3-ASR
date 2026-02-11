import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    import soundfile as sf
except ImportError as exc:
    raise SystemExit("Missing dependency: soundfile. Install with `pip install soundfile`.") from exc

try:
    from pynput.keyboard import Controller, Key, Listener
except ImportError as exc:
    raise SystemExit("Missing dependency: pynput. Install with `pip install pynput`.") from exc

from qwen_asr import Qwen3ASRModel


@dataclass
class AppState:
    recording: bool = False
    last_toggle_ts: float = 0.0


@dataclass
class TranscriptionJob:
    audio: np.ndarray
    wav_path: Optional[Path]
    duration_sec: float


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
        dataset_dir: str,
        wav_subtype: str,
        save_data: bool,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.language = language
        self.dataset_dir = Path(dataset_dir)
        self.save_data = save_data
        if self.save_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.wav_subtype = wav_subtype
        self.manifest_path = self.dataset_dir / "manifest.jsonl"
        self.record_index = 0

        self.state = AppState()
        self.state_lock = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None

        self.jobs: queue.Queue[Optional[TranscriptionJob]] = queue.Queue()
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

    def _next_capture_path(self) -> Path:
        self.record_index += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.dataset_dir / f"capture_{timestamp}_{self.record_index:04d}.wav"

    def _write_manifest(self, wav_path: Path, text: str, language: str, duration_sec: float) -> None:
        item = {
            "audio": str(wav_path),
            "text": text,
            "language": language,
            "sample_rate": self.sample_rate,
            "duration_sec": round(duration_sec, 3),
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _stop_recording(self) -> Optional[TranscriptionJob]:
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
        duration_sec = audio.shape[0] / self.sample_rate
        wav_path: Optional[Path] = None
        if self.save_data:
            wav_path = self._next_capture_path()
            sf.write(str(wav_path), audio, self.sample_rate, subtype=self.wav_subtype)

        print(f"[F8] Recording stopped. Captured {duration_sec:.2f}s audio.")
        if wav_path is not None:
            print(f"Saved WAV: {wav_path}")
        return TranscriptionJob(audio=audio, wav_path=wav_path, duration_sec=duration_sec)

    def _paste_text(self, text: str) -> None:
        if not text:
            print("Empty transcription. Nothing pasted.")
            return

        previous_clipboard = None
        has_previous = False
        try:
            previous_clipboard = pyperclip.paste()
            has_previous = True
        except Exception:  # noqa: BLE001
            has_previous = False

        pyperclip.copy(text)
        time.sleep(0.05)
        with self.keyboard.pressed(Key.ctrl):
            self.keyboard.press("v")
            self.keyboard.release("v")
        time.sleep(0.05)

        if has_previous:
            try:
                pyperclip.copy(previous_clipboard)
            except Exception:  # noqa: BLE001
                pass

        print("Transcription pasted at cursor and clipboard restored.")

    def _worker_loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                if job.wav_path is not None:
                    print(f"Transcribing {job.wav_path.name} ...")
                else:
                    print("Transcribing ...")
                result = self.model.transcribe(
                    audio=(job.audio, self.sample_rate),
                    language=self.language,
                )
                text = result[0].text.strip() if result else ""
                detected_lang = result[0].language if result else "unknown"
                print(f"Detected language: {detected_lang}")
                print(f"Text: {text}")

                if self.save_data and job.wav_path is not None:
                    txt_path = job.wav_path.with_suffix(".txt")
                    txt_path.write_text(text, encoding="utf-8")
                    self._write_manifest(job.wav_path, text, detected_lang, job.duration_sec)
                    print(f"Saved TXT: {txt_path}")
                    print(f"Appended manifest: {self.manifest_path}")
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
                job = self._stop_recording()
                if job is not None:
                    self.jobs.put(job)
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
    parser.add_argument("--dataset-dir", default="dataset_collect", help="Directory for collected wav/text data")
    parser.add_argument(
        "--wav-subtype",
        default="PCM_24",
        choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
        help="Subtype for saved WAV files",
    )
    parser.add_argument(
        "--enable-save-data",
        action="store_true",
        help="Enable saving WAV/TXT/manifest for dataset collection.",
    )
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
        dataset_dir=args.dataset_dir,
        wav_subtype=args.wav_subtype,
        save_data=args.enable_save_data,
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
