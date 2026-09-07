import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from astra.inference import drain_stream, on_executor


@pytest.mark.asyncio
async def test_cancel_waits_for_native_inference_before_releasing_model():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def inference():
        started.set()
        release.wait(timeout=5)
        finished.set()
        return "audio"

    with ThreadPoolExecutor(max_workers=1) as executor:
        task = asyncio.create_task(on_executor(executor, inference))
        await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()


@pytest.mark.asyncio
async def test_native_exception_is_visible():
    def fail():
        raise RuntimeError("Model failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="Model failed"):
            await on_executor(executor, fail)


def test_abandoned_tts_stream_reaches_normal_join_before_reuse():
    joined = []

    def threaded_stream():
        yield b"first audio"
        yield b"remaining audio"
        joined.append(True)  # Represents Pocket TTS's join after its yield loop.

    stream = threaded_stream()
    next(stream)
    drain_stream(stream)
    assert joined == [True]
