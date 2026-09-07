"""Exercise actual WebRTC with synthetic German mic audio, never the real mic."""

import asyncio
import json
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright, expect
from scipy.io.wavfile import write

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / ".runtime"


def make_fixture():
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language="german")
    voice = model.get_state_for_audio_prompt("alba")
    samples = model.generate_audio(voice, "Hallo Astra. Wie heißt du?").numpy()
    # The initial silence lets the pipeline finish establishing WebRTC.
    audio = np.concatenate([np.zeros(24000 * 15), samples, np.zeros(24000 * 35)])
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "test-microphone.wav"
    write(path, 24000, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return path


async def main():
    fixture = ARTIFACTS / "test-microphone.wav"
    if not fixture.exists():
        fixture = await asyncio.to_thread(make_fixture)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fixture}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto("http://localhost:7860")
        await expect(page.locator("#connect")).to_be_enabled(timeout=120000)
        await page.screenshot(path=str(ARTIFACTS / "desktop.png"))
        await page.get_by_role("button", name="Gespräch starten").click()
        try:
            await page.locator(".message.assistant").wait_for(timeout=60000)
            transcript = await page.locator("#messages").inner_text()
            audio_stats = await page.evaluate("""async () => {
                const reports = await peer.getStats();
                return [...reports.values()].filter(report =>
                    report.type === 'inbound-rtp' && report.kind === 'audio'
                ).map(report => ({
                    packets: report.packetsReceived,
                    energy: report.totalAudioEnergy,
                    duration: report.totalSamplesDuration,
                }));
            }""")
            assert any(report.get("energy", 0) > 0 for report in audio_stats), audio_stats
            assert await page.evaluate("!output.paused && output.currentTime > 0")
            await page.get_by_role("button", name="Mikrofon pausieren").click()
            assert await page.locator("#mute").get_attribute("aria-pressed") == "true"
            await page.get_by_role("button", name="Mikrofon aktivieren").click()
            await page.get_by_role("button", name="Gespräch beenden").click()
            assert await page.locator("#mute").is_hidden()
            await page.get_by_role("button", name="Verlauf leeren").click()
            assert await page.locator(".message").count() == 0
            await page.set_viewport_size({"width": 390, "height": 844})
            await page.screenshot(path=str(ARTIFACTS / "mobile.png"))
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors, errors
            print(
                json.dumps(
                    {"passed": True, "transcript": transcript,
                     "audio": audio_stats, "js_errors": errors},
                    ensure_ascii=False,
                )
            )
        finally:
            await page.screenshot(path=str(ARTIFACTS / "last-test.png"))
            print("Browser status:", await page.locator("#status").inner_text())
            print("Browser error:", await page.locator("#error").inner_text())
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
