"""Thread-confined model inference. No audio is stored on disk."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from astra.core import Settings


async def on_executor(executor, function, *args):
    """Finish an in-flight native call before allowing its owner to be reused."""
    future = asyncio.get_running_loop().run_in_executor(executor, function, *args)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await asyncio.shield(future)
        raise


def drain_stream(stream):
    """Let Pocket TTS join its internal threads before reusing the model.

    Its upstream generator only joins on normal exhaustion, not generator.close().
    Discard the unplayed remainder while the audio transport stops immediately.
    """
    for _ in stream:
        pass


class Models:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="astra-stt")
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="astra-tts")
        self.stt = None
        self.tts = None
        self.voice_cache: dict[str, dict] = {}

    def load_stt(self):
        from mlx_audio.stt import load

        self.stt = load(self.settings.stt_model)
        # Ensure this installed MLX version really exposes incremental microphone APIs.
        from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram  # noqa: F401
        from mlx_audio.stt.models.nemotron_asr.streaming import (
            ConformerStreamingState,  # noqa: F401
        )

    def load_tts(self):
        from pocket_tts import TTSModel

        self.tts = TTSModel.load_model(language=self.settings.tts_language)

    def get_voice(self, name: str) -> dict:
        """Compute (or reuse) one voice's conditioning state. Runs on tts_executor."""
        if name not in self.voice_cache:
            self.voice_cache[name] = self.tts.get_state_for_audio_prompt(name)
        return self.voice_cache[name]

    async def close(self):
        await asyncio.to_thread(self.stt_executor.shutdown, wait=True, cancel_futures=True)
        await asyncio.to_thread(self.tts_executor.shutdown, wait=True, cancel_futures=True)


class Recognizer:
    """One utterance's incremental mel, encoder and RNN-T decoder state.

    All methods run on Models.stt_executor, including construction. Encoder and
    decoder caches survive microphone chunks and are discarded at utterance end.
    """

    def __init__(self, model):
        from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
        from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState

        self.model = model
        self.mel = StreamingLogMelSpectrogram(model.preprocessor_config)
        self.encoder = ConformerStreamingState(model.encoder, att_context_size=[56, 3])
        self.last_token = model.blank_id
        self.hidden = None
        self.tokens = []
        self.closed = False

    def push(self, pcm: bytes, final: bool = False) -> str:
        import mlx.core as mx
        from mlx_audio.stt.models.nemotron_asr import tokenizer

        if self.closed:
            raise RuntimeError("Utterance already closed")
        samples = mx.array(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
        mel = self.mel.push(samples, final=final)
        for encoded in self.encoder.push(mel, final=final):
            prompted = self.model.apply_prompt(encoded, "de-DE")
            self.encoder.materialize(prompted)
            for frame_index in range(prompted.shape[1]):
                feature = prompted[:, frame_index : frame_index + 1]
                for _ in range(self.model.max_symbols or 10):
                    token = (
                        mx.array([[self.last_token]], dtype=mx.int32)
                        if self.last_token != self.model.blank_id
                        else None
                    )
                    output, (h, c) = self.model.decoder(token, self.hidden)
                    prediction = int(
                        mx.argmax(self.model.joint(feature, output.astype(feature.dtype)))
                    )
                    if prediction == self.model.blank_id:
                        break
                    self.last_token = prediction
                    self.hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
                    mx.eval(*self.hidden)
                    if not tokenizer.is_special_token(prediction, self.model.vocabulary):
                        self.tokens.append(prediction)
        self.closed = final
        return tokenizer.decode(self.tokens, self.model.vocabulary).strip()
