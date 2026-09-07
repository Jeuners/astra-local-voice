import json
from unittest.mock import patch

import httpx
import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from astra.core import Settings
from astra.services import NativeOllamaService


@pytest.mark.asyncio
async def test_native_stream_sends_think_false_and_returns_content():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text='{"message":{"content":"Hallo"}}\n{"done":true}\n')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = NativeOllamaService(Settings(), lambda message: None)
    with patch("astra.services.httpx.AsyncClient", return_value=client):
        stream = await service.get_chat_completions(LLMContext([{"role": "user", "content": "Hi"}]))
        chunks = [chunk async for chunk in stream]
    assert requests[0]["think"] is False
    assert chunks[0].choices[0].delta.content == "Hallo"
    await service._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        '{"message":{"thinking":"secret reasoning"}}\n',
        '{"error":"out of memory"}\n',
        '{"message":{"content":"incomplete"}}\n',
    ],
)
async def test_bad_ollama_streams_fail_explicitly(body):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    )
    service = NativeOllamaService(Settings(), lambda message: None)
    with patch("astra.services.httpx.AsyncClient", return_value=client):
        stream = await service.get_chat_completions(LLMContext([{"role": "user", "content": "Hi"}]))
        with pytest.raises(RuntimeError):
            _ = [chunk async for chunk in stream]
    await service._client.close()
