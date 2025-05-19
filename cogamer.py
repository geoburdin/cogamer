import faulthandler; faulthandler.enable()

import os
import asyncio
import base64
import io
import traceback
import audioop
import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types
import dotenv
from google.genai.types import RealtimeInputConfig, AutomaticActivityDetection

dotenv.load_dotenv()
FORMAT = pyaudio.paInt16
SEND_RATE     = 16000   # for Gemini
MIC_CHANNELS  = 1
pya = pyaudio.PyAudio()

# detect native rates dynamically
input_info  = pya.get_default_input_device_info()
output_info = pya.get_default_output_device_info()

NATIVE_RATE  = int(input_info['defaultSampleRate'])
RECEIVE_RATE = 24000

# pick a ~40 ms chunk size
CHUNK_NATIVE = int(NATIVE_RATE * 0.04)
MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "screen"

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools = [
    types.Tool(code_execution=types.ToolCodeExecution()),
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(
      function_declarations=[
        types.FunctionDeclaration(
          name="click_on",
          description="Move the mouse to given (x,y) and click",
          parameters={
            "type": "object",
            "properties": {
              "x":   {"type": "number"},
              "y":   {"type": "number"},
              "clicks":   {"type": "integer", "default": 1},
              "interval": {"type": "number",  "default": 0.0},
              "button":   {"type": "string",  "default": "left"}
            },
            "required": ["x","y"]
          }
        )
      ]
    )
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="You are professional Games QA Engineer, and there is a stream being sent to you in form of the separate frames. ANSWER ONLY short, dont be talkative")],
        role="user"
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        # When the total context reaches ~20 000 tokens, compress the oldest turns
        trigger_tokens=20_000,
        sliding_window=types.SlidingWindow(
            # After compression, keep the most recent 10 000 tokens uncompressed
            target_tokens=10_000
        )
    ),
    realtime_input_config = RealtimeInputConfig(automatic_activity_detection =  AutomaticActivityDetection(
            disabled= False,
            start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_HIGH,
            end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_LOW,
            prefix_padding_ms = 25,
            silence_duration_ms = 100)
    )
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if msg.get("mime_type", "").startswith("audio"):
                await self.session.send_realtime_input(audio=msg)
            else:
                await self.session.send_realtime_input(video=msg)

    async def listen_audio(self):
        mic = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=MIC_CHANNELS,
            rate=NATIVE_RATE,  # dynamic native rate
            input=True,
            frames_per_buffer=CHUNK_NATIVE,
        )
        while True:
            native = await asyncio.to_thread(
                mic.read,
                CHUNK_NATIVE,
                exception_on_overflow=False
            )
            pcm16k, _ = audioop.ratecv(  # down-sample once per 43 ms block
                native,  # bytes
                2,  # 2 bytes per sample (Int16)
                MIC_CHANNELS,
                NATIVE_RATE,
                SEND_RATE,
                None
            )
            await self.out_queue.put({
                "data": pcm16k,
                "mime_type": "audio/pcm"
            })

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        spk = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=1,
            rate=NATIVE_RATE,  # dynamic native rate
            output=True,
        )
        while True:
            pcm24 = await self.audio_in_queue.get()
            # convert from 24k → native
            pcm_native, _ = audioop.ratecv(
                pcm24, 2, 1,
                RECEIVE_RATE, NATIVE_RATE,
                None
            )
            await asyncio.to_thread(spk.write, pcm_native)

    async def run(self):
        while True:
            try:
                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    await send_text_task
                    raise asyncio.CancelledError("User requested exit")

            except Exception as e:
                if hasattr(self, "audio_stream") and self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                print(f"[run] session crashed: {e}\n")
                await asyncio.sleep(2)
                continue

import pyautogui
def click_on(x, y, clicks=1, interval=0.0, button='left'):
    """
    Moves the mouse to (x,y) and clicks.
    """
    pyautogui.moveTo(x, y)
    pyautogui.click(clicks=clicks, interval=interval, button=button)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
