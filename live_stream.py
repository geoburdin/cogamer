# -*- coding: utf-8 -*-
import asyncio
import base64
import json
import io
import os
import sys
import traceback
import pyaudio
import PIL.Image
import mss
import mss.tools
import dotenv
dotenv.load_dotenv()

from websockets.asyncio.client import connect
from langsmith import traceable

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

# Model and API settings
HOST = "generativelanguage.googleapis.com"
MODEL = "models/gemini-2.0-flash-live-001"
API_KEY = os.environ.get("GEMINI_API_KEY")
URI = f"wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={API_KEY}"

class GlobalContext:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.custom_state = {}

    @traceable
    def add_message(self, role: str, text: str):
        self.conversation_history.append((role, text))

    @traceable
    def get_history(self):
        return self.conversation_history

    @traceable
    def set_preference(self, key: str, value):
        self.user_preferences[key] = value

    @traceable
    def get_preference(self, key: str, default=None):
        return self.user_preferences.get(key, default)

    def to_json(self):
        return {
            "conversation_history": self.conversation_history,
            "user_preferences": self.user_preferences,
            "custom_state": self.custom_state,
        }


global_context = GlobalContext()

from google.genai import types

@traceable
async def save_user_preferences(text):
    with open('user_preferences.txt', 'w') as f:
        f.write(text)

async def handle_tool_call(ws, tool_call):
    """Handles incoming tool calls from Gemini and sends the appropriate response."""
    print("Handling tool call:", tool_call)

    function_name = tool_call["functionCalls"][0]["name"]
    arguments = tool_call["functionCalls"][0]["args"]

    if function_name == "save_user_preferences":
        await save_user_preferences(str(global_context.to_json()))
        response = "User preferences saved to user_preferences.txt."
    elif function_name == "remember_user_preferences":
        key = arguments.get("key")
        value = arguments.get("value")
        global_context.user_preferences[key] = value
        response = f"Preference {key} set to {value}."
    msg = {
        'tool_response': {
            'function_responses': [{
                'id': tool_call["functionCalls"][0]['id'],
                'name': function_name,
                'response':{'result': {'string_value': response}}
            }]
        }
    }

    # Send the tool response back to the server
    await ws.send(json.dumps(msg))
@traceable
def handle_server_content(wf, server_content):
    # server_content is a dict, so handle keys with get
    if not server_content:
        return

    model_turn = server_content.get('modelTurn')
    if model_turn:
        parts = model_turn.get('parts', [])
        for part in parts:
            executable_code = part.get('executableCode')
            if executable_code is not None:
                print("-------------------------------")
                print("``` python")
                print(executable_code.get('code', ''))
                print("```")
                print("-------------------------------")

            code_execution_result = part.get('codeExecutionResult')
            if code_execution_result is not None:
                print("-------------------------------")
                print("```")
                print(code_execution_result.get('output', ''))
                print("```")
                print("-------------------------------")

    # If there is grounding metadata
    grounding_metadata = server_content.get('groundingMetadata')
    if grounding_metadata:
        # Handle grounding metadata if needed.
        pass


class Agent:
    def __init__(self, global_context: GlobalContext):
        self.global_context = global_context
        self.ws = None
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None

    @traceable
    async def startup(self, tools):
        setup_msg = {"setup": {"model": f"models/{MODEL}", "tools": tools}}
        await self.ws.send(json.dumps(setup_msg))
        setup_response = json.loads(await self.ws.recv())


    @traceable
    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            self.global_context.add_message("user", text)
            msg = {
                "client_content": {
                    "turn_complete": True,
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                }
            }
            await self.ws.send(json.dumps(msg))

    def _capture_screen_frame(self):
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_encoded = base64.b64encode(image_io.read()).decode()
        return {"mime_type": "image/jpeg", "data": image_encoded}

    @traceable
    async def stream_screen_frames(self, interval: float = 1.0):
        while True:
            frame = await asyncio.to_thread(self._capture_screen_frame)
            msg = {"realtime_input": {"media_chunks": [frame]}}
            await self.out_queue.put(msg)
            await asyncio.sleep(interval)

    @traceable
    async def listen_audio(self):
        pya_in = pyaudio.PyAudio()
        mic_info = pya_in.get_default_input_device_info()
        self.audio_stream = pya_in.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE)
            msg = {
                "realtime_input": {
                    "media_chunks": [
                        {
                            "data": base64.b64encode(data).decode(),
                            "mime_type": "audio/pcm",
                        }
                    ]
                }
            }
            await self.out_queue.put(msg)

    @traceable
    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.ws.send(json.dumps(msg))

    @traceable
    async def receive_audio(self):
        async for raw_response in self.ws:
            response = json.loads(raw_response)

            inline_data = (
                response
                .get("serverContent", {})
                .get("modelTurn", {})
                .get("parts", [{}])[0]
                .get("inlineData", {})
                .get("data")
            )
            if inline_data:
                pcm_data = base64.b64decode(inline_data)
                self.audio_in_queue.put_nowait(pcm_data)

            turn_complete = response.get("serverContent", {}).get("turnComplete", False)
            if turn_complete:
                print("\nEnd of turn")
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

            tool_call = response.get('toolCall')
            if tool_call is not None:
                await handle_tool_call(self.ws, tool_call)

            server_content = response.get('serverContent')
            if server_content:
                handle_server_content(None, server_content)

    @traceable
    async def play_audio(self):
        pya_out = pyaudio.PyAudio()
        stream = pya_out.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True
        )

        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    @traceable
    async def run(self):
        try:


            async with (
                await connect(URI, additional_headers={"Content-Type": "application/json"}) as ws,
                asyncio.TaskGroup() as tg,
            ):
                self.ws = ws
                tools_custom = [

                    {
                        "name": "save_user_preferences",
                        "description": "Saves user preferences to a file."
                    },
                    {
                        "name": "remember_user_preferences",
                        "description": "Store user preferences in memory.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Preference name."
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Preference value."
                                }
                            },
                            "required": ["key", "value"]
                        }
                    }
                ]
                await self.startup(tools=[
                    {'function_declarations': tools_custom}
                ])

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.stream_screen_frames())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    agent = Agent(global_context=global_context)
    asyncio.run(agent.run())