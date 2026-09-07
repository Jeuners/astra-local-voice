import unittest

from astra.core import Settings, build_request, local_origin_allowed, trim_messages


class CoreTests(unittest.TestCase):
    def test_thinking_is_disabled_in_every_native_request(self):
        request = build_request(Settings(), [{"role": "user", "content": "Hallo"}])
        self.assertIs(request["think"], False)
        self.assertIs(request["stream"], True)
        self.assertEqual(request["model"], "qwen3.5:latest")
        self.assertNotIn("think", request["options"])
        self.assertEqual(request["options"]["num_ctx"], 4096)

    def test_history_stays_bounded_and_preserves_system_and_latest_user(self):
        messages = [{"role": "system", "content": "Deutsch"}]
        for i in range(40):
            messages.extend(
                [
                    {"role": "user", "content": f"Frage {i}"},
                    {"role": "assistant", "content": "Antwort " * 100},
                ]
            )
        messages.append({"role": "user", "content": "Neueste Frage"})
        trimmed = trim_messages(messages, max_chars=5000)
        self.assertEqual(trimmed[0], messages[0])
        self.assertEqual(trimmed[1]["role"], "user")
        self.assertEqual(trimmed[-1], messages[-1])
        self.assertLessEqual(sum(len(m["content"]) for m in trimmed), 5000)
        self.assertEqual(len(messages), 82)

    def test_rejects_foreign_web_origins(self):
        self.assertTrue(local_origin_allowed("http://localhost:7860"))
        self.assertTrue(local_origin_allowed("http://127.0.0.1:7860"))
        self.assertFalse(local_origin_allowed("https://evil.example"))
        self.assertFalse(local_origin_allowed("http://localhost.evil.example:7860"))
        self.assertFalse(local_origin_allowed("null"))

    def test_huge_last_message_is_bounded(self):
        trimmed = trim_messages(
            [
                {"role": "system", "content": "Deutsch"},
                {"role": "user", "content": "x" * 20000},
            ],
            max_chars=5000,
        )
        self.assertLessEqual(sum(len(m["content"]) for m in trimmed), 5000)


if __name__ == "__main__":
    unittest.main()
