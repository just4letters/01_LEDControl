"""
core/voice.py

Centralized Whisper listener thread with optional Gemini Intent Parsing.
Supports structured JSON parsing with model fallbacks to guarantee reliability.
"""

import speech_recognition as sr
import threading
import time
import queue
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from core import config
from core.mic import open_microphone

load_dotenv()

# Global Client setup
gemini_client = None
if os.getenv("GEMINI_API_KEY"):
    try:
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"⚠️ Voice AI warning: Could not initialize Gemini client: {e}")


class VoiceListener:
    def __init__(self, sentence_queue, action_queue, custom_parser=None, ai_context_prompt=None, response_schema=None):
        """
        :param sentence_queue: Queue to send transcribed text to external worker loops.
        :param action_queue: Queue to push resolved actions directly to apps.
        :param custom_parser: Optional fast-path Python function.
        :param ai_context_prompt: Base instructions string for Gemini.
        :param response_schema: Optional types.Schema for structured outputs.
        """
        self.sentence_queue = sentence_queue
        self.action_queue = action_queue
        self.custom_parser = custom_parser
        self.ai_context_prompt = ai_context_prompt
        self.response_schema = response_schema
        self.running = False

    def start(self, prompt_message="[VOICE] Listener active..."):
        self.running = True
        threading.Thread(target=self._listen_loop, args=(prompt_message,), daemon=True).start()

    def stop(self):
        self.running = False

    def _resolve_intent_with_gemini(self, text):
        if not gemini_client or not self.ai_context_prompt:
            return None

        # Updated active Gemini models
        fallback_models = ['gemini-2.5-flash-lite', 'gemini-flash-lite-latest', 'gemini-2.5-pro']

        # Evaluate prompt function if callable
        prompt = self.ai_context_prompt() if callable(self.ai_context_prompt) else self.ai_context_prompt

        gen_config_kwargs = {"response_mime_type": "application/json"}
        if self.response_schema is not None:
            gen_config_kwargs["response_schema"] = self.response_schema

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)

        for model_name in fallback_models:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=f"{prompt}\nUser Spoke: \"{text}\"",
                    config=gen_config,
                )
                payload = json.loads(response.text)
                return payload
            except Exception as e:
                continue

        return None

    def _listen_loop(self, prompt_message):
        print("[DEBUG] VoiceListener thread started.")
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = config.MIC_ENERGY_THRESHOLD
        recognizer.dynamic_energy_threshold = False

        while self.running:
            try:
                with open_microphone(sample_rate=16000) as source:
                    print(f"\n{prompt_message}")
                    while self.running:
                        try:
                            audio = recognizer.listen(source, phrase_time_limit=5)
                            text = recognizer.recognize_whisper(audio, model="base.en").strip()

                            if text:
                                text_lower = text.lower()
                                print(f"\n[🎤 Heard]: '{text}'")

                                # 1. Universal Kill Switch
                                kill_words = ["shut down", "shutdown", "quit", "exit", "close app"]
                                if any(word in text_lower for word in kill_words):
                                    self.action_queue.put({"action": "quit"})
                                    return
                                
                                # 2. Universal Clear Screen
                                clear_words = ["clear screen", "clear the screen", "wipe screen"]
                                if any(word in text_lower for word in clear_words):
                                    self.action_queue.put({"action": "clear_screen"})
                                    continue # Skip AI/custom parser since we already resolved the command

                                # 3. App-Specific Fast Path (Python parser)
                                if self.custom_parser:
                                    custom_action = self.custom_parser(text_lower)
                                    if custom_action:
                                        self.action_queue.put(custom_action)
                                        continue

                                # 4. Gemini Intent Resolution (Fallback-enabled)
                                if self.ai_context_prompt:
                                    ai_action = self._resolve_intent_with_gemini(text)
                                    if ai_action and ai_action.get("action") != "ignore":
                                        print(f"🤖 AI Action: {ai_action}")
                                        self.action_queue.put(ai_action)
                                        continue

                                # 5. Sentence Queue Push (For app-level workers)
                                while not self.sentence_queue.empty():
                                    try:
                                        self.sentence_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                self.sentence_queue.put(text)

                        except sr.WaitTimeoutError:
                            continue
                        except sr.UnknownValueError:
                            pass
                        except Exception as inner_e:
                            print(f"[Whisper Loop Error]: {inner_e}")

            except Exception as outer_e:
                print(f"⚠️ Microphone Error: {outer_e}. Rebooting in 2s...")
                time.sleep(2)