"""Download and warm local models; never opens the microphone."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from astra.core import Settings, build_request
from astra.inference import Models, Recognizer, on_executor


async def main():
    import nltk

    nltk_path = Path(__file__).resolve().parent.parent / ".cache" / "nltk"
    nltk_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not nltk.download("punkt_tab", download_dir=str(nltk_path), quiet=True):
        raise RuntimeError("Satzsegmentierung konnte nicht heruntergeladen werden")
    settings = Settings.from_env()
    models = Models(settings)
    try:
        print("Lade Nemotron für deutsche Streaming-Erkennung …", flush=True)
        await on_executor(models.stt_executor, models.load_stt)
        recognizer = await on_executor(models.stt_executor, Recognizer, models.stt)
        await on_executor(models.stt_executor, recognizer.push, bytes(16000), True)
        print("Nemotron bereit. Lade deutsche Pocket-TTS-Stimme …", flush=True)
        await on_executor(models.tts_executor, models.load_tts)
        voice_state = await on_executor(models.tts_executor, models.get_voice, settings.voice)

        def synthesize():
            start = time.monotonic()
            chunks = []
            first_ms = None
            for chunk in models.tts.generate_audio_stream(
                voice_state,
                "Hallo, ich bin Astra. Ich laufe vollständig auf deinem Mac.",
                copy_state=True,
            ):
                if first_ms is None:
                    first_ms = round((time.monotonic() - start) * 1000)
                chunks.append(chunk.numpy())
            import numpy as np

            samples = np.concatenate(chunks)
            return samples, {
                "first_audio_ms": first_ms,
                "seconds": round(len(samples) / models.tts.sample_rate, 2),
                "generation_ms": round((time.monotonic() - start) * 1000),
            }

        samples, metrics = await on_executor(models.tts_executor, synthesize)
        print(f"Pocket TTS: {json.dumps(metrics)}", flush=True)
        import numpy as np
        from scipy.signal import resample_poly

        audio16 = resample_poly(samples, 2, 3)
        pcm = (np.clip(audio16, -1, 1) * 32767).astype("<i2").tobytes()
        recognizer = await on_executor(models.stt_executor, Recognizer, models.stt)
        started = time.monotonic()
        partials = []
        for offset in range(0, len(pcm), 10240):
            text = await on_executor(
                models.stt_executor, recognizer.push, pcm[offset : offset + 10240], False
            )
            if text and (not partials or text != partials[-1]):
                partials.append(text)
        text = await on_executor(models.stt_executor, recognizer.push, b"", True)
        print(f"ASR-Rücktest: {text}", flush=True)
        print(
            f"ASR: {len(partials)} Live-Zwischenstände, {round((time.monotonic() - started) * 1000)} ms",
            flush=True,
        )
        if "astra" not in text.lower() or "mac" not in text.lower():
            raise RuntimeError("Deutscher Sprach-Rücktest fehlgeschlagen")
        async with httpx.AsyncClient(timeout=180) as client:
            payload = build_request(
                settings, [{"role": "user", "content": "Sage kurz Hallo auf Deutsch."}]
            )
            response = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            events = [json.loads(line) for line in response.text.splitlines() if line]
            if any(e.get("message", {}).get("thinking") for e in events):
                raise RuntimeError("Thinking ist nicht deaktiviert")
            answer = "".join(e.get("message", {}).get("content", "") for e in events)
            if not answer or not events[-1].get("done"):
                raise RuntimeError("Ollama hat keine vollständige Antwort geliefert")
            print(f"Ollama ohne Thinking: {answer}", flush=True)
        print("Alle drei lokalen Modelle erfolgreich geprüft.", flush=True)
    finally:
        await models.close()


if __name__ == "__main__":
    asyncio.run(main())
