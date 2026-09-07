# astra

Ein lokaler deutscher Sprachagent für Apple Silicon. Kein Cloud-Anruf, kein
Tracking, kein gespeichertes Audio — Spracherkennung, Sprachmodell und
Sprachausgabe laufen alle auf deinem Mac.

## Wie es funktioniert

Drei lokale Modelle, verbunden über eine [Pipecat](https://github.com/pipecat-ai/pipecat)-Pipeline:

| Stufe | Modell | Wo |
|---|---|---|
| Spracherkennung | Nemotron ASR (streaming) | MLX, on-device |
| Sprachmodell | Qwen 3.5 | über natives Ollama `/api/chat` |
| Sprachausgabe | Pocket TTS, Stimme „alba" | MLX, on-device |

Der Browser spricht per WebRTC direkt mit einem FastAPI-Server auf
`localhost:7860`. Der Server ist bewusst nur lokal erreichbar: Host- und
Origin-Prüfung auf jedem Request, strikte Content-Security-Policy, keine
offenen Ports nach außen.

## Starten

```bash
uv sync
uv run python -m astra.prepare   # lädt & prüft alle drei Modelle einmalig
uv run python -m astra.server    # startet auf http://localhost:7860
```

Ollama muss separat laufen (`ollama serve`) und `qwen3.5:latest` muss
gezogen sein. Die Seite öffnen, Mikrofon erlauben, sprechen.

## Konfiguration

Über Umgebungsvariablen, siehe `astra/core.py::Settings`:

| Variable | Default |
|---|---|
| `ASTRA_MODEL` | `qwen3.5:latest` |
| `ASTRA_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `ASTRA_STT_MODEL` | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` |
| `ASTRA_TTS_LANGUAGE` | `german` |
| `ASTRA_VOICE` | `alba` |
| `ASTRA_PORT` | `7860` |

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Sicherheit

- Nur `localhost`/`127.0.0.1` erreichbar, alle anderen Hosts bekommen 403
- POST-Requests werden gegen den erwarteten Origin geprüft
- `think` ist im Ollama-Request hart auf `false` gesetzt — die Pipeline
  wirft, falls das Modell trotzdem Denkausgabe liefert
- Kein Audio, keine Transkripte werden auf Disk geschrieben; der
  Gesprächsverlauf lebt nur im Speicher der laufenden Session

---

Erstellt von Astra (Grunddeploy), gecheckt, dokumentiert und bewertet durch Claude.
