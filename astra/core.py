"""Configuration and pure request policy, independent of audio hardware."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "qwen3.5:latest"
    ollama_url: str = "http://127.0.0.1:11434"
    stt_model: str = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    tts_language: str = "german"
    voice: str = "alba"
    port: int = 7860
    context_tokens: int = 4096
    tailnet_host: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            model=os.getenv("ASTRA_MODEL", cls.model),
            ollama_url=os.getenv("ASTRA_OLLAMA_URL", cls.ollama_url).rstrip("/"),
            stt_model=os.getenv("ASTRA_STT_MODEL", cls.stt_model),
            tts_language=os.getenv("ASTRA_TTS_LANGUAGE", cls.tts_language),
            voice=os.getenv("ASTRA_VOICE", cls.voice),
            port=int(os.getenv("ASTRA_PORT", cls.port)),
            tailnet_host=os.getenv("ASTRA_TAILNET_HOST", cls.tailnet_host),
        )


VOICES = (
    {"name": "alba", "display_name": "Alba", "gender": "weiblich"},
    {"name": "anna", "display_name": "Anna", "gender": "weiblich"},
    {"name": "azelma", "display_name": "Azelma", "gender": "weiblich"},
    {"name": "bill_boerst", "display_name": "Bill Boerst", "gender": "männlich"},
    {"name": "caro_davy", "display_name": "Caro Davy", "gender": "weiblich"},
    {"name": "charles", "display_name": "Charles", "gender": "männlich"},
    {"name": "cosette", "display_name": "Cosette", "gender": "weiblich"},
    {"name": "eponine", "display_name": "Eponine", "gender": "weiblich"},
    {"name": "estelle", "display_name": "Estelle", "gender": "weiblich"},
    {"name": "eve", "display_name": "Eve", "gender": "weiblich"},
    {"name": "fantine", "display_name": "Fantine", "gender": "weiblich"},
    {"name": "george", "display_name": "George", "gender": "männlich"},
    {"name": "giovanni", "display_name": "Giovanni", "gender": "männlich"},
    {"name": "jane", "display_name": "Jane", "gender": "weiblich"},
    {"name": "javert", "display_name": "Javert", "gender": "männlich"},
    {"name": "jean", "display_name": "Jean", "gender": "männlich"},
    {"name": "juergen", "display_name": "Juergen", "gender": "männlich"},
    {"name": "lola", "display_name": "Lola", "gender": "weiblich"},
    {"name": "marius", "display_name": "Marius", "gender": "männlich"},
    {"name": "mary", "display_name": "Mary", "gender": "weiblich"},
    {"name": "michael", "display_name": "Michael", "gender": "männlich"},
    {"name": "paul", "display_name": "Paul", "gender": "männlich"},
    {"name": "peter_yearsley", "display_name": "Peter Yearsley", "gender": "männlich"},
    {"name": "rafael", "display_name": "Rafael", "gender": "männlich"},
    {"name": "stuart_bell", "display_name": "Stuart Bell", "gender": "männlich"},
    {"name": "vera", "display_name": "Vera", "gender": "weiblich"},
)
VOICE_NAMES = frozenset(voice["name"] for voice in VOICES)


SYSTEM_PROMPT = (
    "Du bist Astra, ein freundlicher deutschsprachiger Gesprächsassistent. "
    "Antworte natürlich und knapp, normalerweise in ein bis drei kurzen Sätzen. "
    "Deine Antwort wird vorgelesen: kein Markdown, keine Sternchen, keine Listen. "
    "Sprich Zahlen und Abkürzungen verständlich aus. Stelle bei Bedarf eine kurze Rückfrage. "
    "Du hast keine Werkzeuge, keinen Internetzugang und keinen Zugriff auf Dateien oder Apps. "
    "Behaupte nicht, Aktionen ausgeführt zu haben."
)


def trim_messages(messages: list[dict], max_chars: int = 10000) -> list[dict]:
    """Retain recent whole turns within a conservative context character budget."""
    system = [dict(m) for m in messages if m["role"] == "system"][:1]
    if system:
        system[0]["content"] = system[0]["content"][: max_chars // 2]
    budget = max_chars - sum(len(m["content"]) for m in system)
    turns = []
    for message in reversed(messages):
        if message["role"] not in ("user", "assistant"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        if len(content) > budget:
            if not turns:
                turns.append({"role": message["role"], "content": content[-budget:]})
            break
        turns.append({"role": message["role"], "content": content})
        budget -= len(content)
    turns.reverse()
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return system + turns


def build_request(settings: Settings, messages: list[dict]) -> dict:
    return {
        "model": settings.model,
        "messages": trim_messages(messages),
        "think": False,
        "stream": True,
        "keep_alive": -1,
        "options": {
            "num_ctx": settings.context_tokens,
            "num_predict": 256,
            "temperature": 0.6,
        },
    }


def local_origin_allowed(origin: str, port: int = 7860, tailnet_host: str | None = None) -> bool:
    allowed = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
    if tailnet_host:
        allowed.add(f"https://{tailnet_host}")
    return origin in allowed
