from unittest.mock import AsyncMock, patch

import pytest
from pipecat.processors.frame_processor import FrameDirection

from astra.core import Settings
from astra.inference import Models
from astra.services import NemotronSTTService


@pytest.mark.asyncio
async def test_recognition_failure_reaches_pipeline_upstream_and_notifies_browser():
    models = Models(Settings())
    notices = []
    service = NemotronSTTService(models, notices.append)
    service.push_frame = AsyncMock()
    try:
        service.queue.put_nowait(("start", b""))
        with patch("astra.services.Recognizer", side_effect=RuntimeError("Model failed")):
            await service._worker()
        assert service.failed
        assert notices[0]["type"] == "error"
        assert "Model failed" in notices[0]["message"]
        frame, direction = service.push_frame.call_args.args
        assert frame.fatal
        assert direction == FrameDirection.UPSTREAM
    finally:
        await models.close()
