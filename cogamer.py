# cogamer.py
import datetime
import asyncio
import base64
import json
import io
import os
import sys
import traceback
import random
import pyaudio
import PIL.Image
import mss
import mss.tools
import dotenv
import logging

dotenv.load_dotenv()

from websockets.asyncio.client import connect
from langsmith import traceable
from typing import List, Dict
from langchain_openai import ChatOpenAI
from schemas import FrameAnalysis, Context, DetectGameFocusPoints
from langchain_core.messages import HumanMessage

# -----------------------------
# Configuration
# -----------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

# Model and API settings
HOST = "generativelanguage.googleapis.com"
MODEL = "gemini-2.0-flash-exp"
API_KEY = os.environ.get("GEMINI_API_KEY")
URI = f"wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={API_KEY}"

# LangChain Model Setup for Frame Analysis
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
structured_llm_frame_analysis = model.with_structured_output(FrameAnalysis)
structured_llm_detect_game_focus_points = model.with_structured_output(DetectGameFocusPoints)

# -----------------------------
# Global Context Class
# -----------------------------

class GlobalContext:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.custom_state = {}
        self.game = "Unknown"
        self.category = "Speedrun Analysis"
        self.focus_points = []
        self.notes = []
        self.frame_analysis_results = []

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

    @traceable
    def to_json(self):
        return {
            "conversation_history": self.conversation_history,
            "user_preferences": self.user_preferences,
            "custom_state": self.custom_state,
            "game": self.game,
            "category": self.category,
            "focus_points": self.focus_points,
            "notes": self.notes,
            "frame_analysis_results": self.frame_analysis_results
        }

global_context = GlobalContext()

# -----------------------------
# Background Frame Analysis
# -----------------------------

@traceable
def detect_game_and_focus_points(frames_data: List[str]) -> Dict:
    """Analyze random frames to detect the game and focus points."""
    logging.info("Analyzing frames to detect game and focus points...")
    message = HumanMessage(
        content=[
                    {
                        "type": "text",
                        "text": """
You are an expert gaming analyst. Analyze the following gameplay frames to detect:
1. The name of the game (if possible).
2. Key focus points to analyze during the video, such as specific tricks, objects important for the gamer community, or strategies.
Be specific, professional, and use gaming terminology.
"""
                    }
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    result = structured_llm_detect_game_focus_points.invoke([message])
    analysis = result.model_dump()
    logging.info(f"Game Detection Result: {analysis}")
    return analysis

@traceable
def analyze_frame(frame_id: int, frames_data: List[str], context: Context) -> Dict:
    """Analyze a batch of frames and return the structured output."""
    logging.info(f"Analyzing frames {frame_id-10}-{frame_id} seconds...")
    message = HumanMessage(
        content=[
                    {
                        "type": "text",
                        "text": f"""
You are a professional speedrun commentator analyzing a gameplay video of '{context.game}' in the category '{context.category}'.
Focus on these key points during analysis: {', '.join(context.focus_points)}.

Provide the following:
1. Frame-specific analysis:
   - Tricks, skips, glitches, and movement optimizations.
   - Mistakes or inefficiencies.
   - Execution precision and routing decisions.
2. New global notes for the speedrun (if applicable).
"""
                    }
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    result = structured_llm_frame_analysis.invoke([message])
    analysis_result = result.model_dump()
    analysis_result["timestamp_id"] = f"{frame_id-10}-{frame_id} seconds"
    logging.info(f"Frame Analysis Result: {analysis_result}")
    return analysis_result

@traceable
def summarize_results(results: List[dict], context: Context) -> str:
    """Summarize the frame analysis results."""
    logging.info("Summarizing analysis results...")
    summary = {
        "total_frames_analyzed": len(results),
        "global_context": context.notes,
        "comments": [res.get("comments", "") for res in results],
        "recommendations": [res.get("recommendations", "") for res in results],
        "tricks_used": [res.get("tricks_used", "") for res in results],
        "good_actions": [res.get("good_actions", "") for res in results],
        "bad_actions": [res.get("bad_actions", "") for res in results]
    }
    summary_json = json.dumps(summary, indent=2)
    logging.info(f"Summary: {summary_json}")
    return summary_json

@traceable
def generate_end_report(summary: str) -> str:
    """Generate an end report summarizing the entire speedrun."""
    logging.info("Generating end report...")
    report_request = f"""
Here is the structured analysis of a gameplay session:

{summary}

Write a detailed report highlighting:
- The overall gameplay and performance
- Key mistakes or inefficiencies
- Best strategies, tricks, and optimizations
- Final recommendations for improvement.
"""
    report_message = HumanMessage(content=[{"type": "text", "text": report_request}])
    result = model.invoke([report_message])
    end_report = result.content
    logging.info("End Report Generated.")
    return end_report

# -----------------------------
# Tool Functions
# -----------------------------

@traceable
async def save_user_preferences(text):
    """Save user preferences to a file."""
    with open('user_preferences.txt', 'w') as f:
        f.write(text)
    logging.info("User preferences saved to 'user_preferences.txt'.")

@traceable
async def remember_user_preferences(key: str, value: str):
    """Store user preferences in memory."""
    global_context.set_preference(key, value)
    logging.info(f"Preference '{key}' set to '{value}'.")

@traceable
async def perform_game_detection(frames_data: List[str]):
    """Perform game detection and update global context."""
    analysis = await asyncio.to_thread(detect_game_and_focus_points, frames_data)
    global_context.game = analysis.get("game", "Unknown")
    global_context.focus_points = analysis.get("focus_points", [])
    logging.info(f"Detected Game: {global_context.game}")
    logging.info(f"Focus Points: {global_context.focus_points}")

async def handle_tool_call(ws, tool_call):
    """Handles incoming tool calls from Gemini and sends the appropriate response."""
    logging.info(f"Handling tool call: {tool_call}")
    function_call = tool_call["functionCalls"][0]
    function_name = function_call["name"]
    arguments = function_call["args"]

    if function_name == "save_user_preferences":
        await save_user_preferences(str(global_context.to_json()))
        response = "User preferences saved to 'user_preferences.txt'."
    elif function_name == "remember_user_preferences":
        key = arguments.get("key")
        value = arguments.get("value")
        await remember_user_preferences(key, value)
        response = f"Preference '{key}' set to '{value}'."
    elif function_name == "perform_game_detection":
        frames = arguments.get("frames", [])
        if frames:
            await perform_game_detection(frames)
            response = f"Game detection performed. Current game: {global_context.game}."
        else:
            response = "No frames provided for game detection."
    else:
        response = f"Function '{function_name}' is not recognized."

    msg = {
        'tool_response': {
            'function_responses': [{
                'id': function_call['id'],
                'name': function_name,
                'response': {'result': {'string_value': response}}
            }]
        }
    }

    await ws.send(json.dumps(msg))
    logging.info(f"Tool response sent for function '{function_name}'.")

# -----------------------------
# Real-time Assistant Class
# -----------------------------

class Agent:
    def __init__(self, global_context: GlobalContext):
        self.global_context = global_context
        self.ws = None
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.collected_frames = []  # Store raw frames as base64 for analysis
        self.frame_counter = 0

    @traceable
    async def startup(self, tools):
        """Initial setup for the WebSocket connection."""
        setup_msg = {"setup": {"model": f"models/{MODEL}", "tools": tools}}
        await self.ws.send(json.dumps(setup_msg))
        setup_response = json.loads(await self.ws.recv())
        logging.info("WebSocket connection established and setup complete.")

    @traceable
    async def send_text(self):
        """Handle user text input and send to the model."""
        while True:
            text = await asyncio.to_thread(input, "You: ")
            if text.lower() == "q":
                # When user quits, generate final report
                await self.generate_final_report()
                await self.ws.close()
                break
            self.global_context.add_message("user", text)

            # Generate a concise summary of recent analysis
            if self.global_context.frame_analysis_results:
                latest_analysis = self.global_context.frame_analysis_results[-1]
                analysis_summary = f"Recent analysis: {json.dumps(latest_analysis, indent=2)}"
            else:
                analysis_summary = "No recent analysis available."

            # Craft the prompt with analysis summary and user message
            prompt = f"""
You are a friendly gaming assistant helping me improve my gameplay in '{self.global_context.game}'.
Based on the current situation on the screen and your analysis of my recent gameplay, provide me with strategic advice and feedback.

{analysis_summary}

User: {text}
Assistant:
"""
            msg = {
                "client_content": {
                    "turn_complete": True,
                    "turns": [{"role": "user", "parts": [{"text": prompt}]}],
                }
            }
            await self.ws.send(json.dumps(msg))
            logging.info("Message sent to assistant.")

    def _capture_screen_frame(self) -> str:
        """Capture a single screen frame and return it as a base64-encoded JPEG."""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail([1024, 1024])  # Resize for efficiency

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_encoded = base64.b64encode(image_io.read()).decode()
        return image_encoded

    @traceable
    async def stream_screen_frames(self, interval: float = 1.0):
        """Continuously capture and send screen frames."""
        while True:
            frame = await asyncio.to_thread(self._capture_screen_frame)
            # Collect frames for periodic analysis
            self.collected_frames.append(frame)
            self.frame_counter += 1

            # Prepare realtime input message
            msg = {"realtime_input": {"media_chunks": [
                {"mime_type": "image/jpeg", "data": frame}
            ]}}
            await self.out_queue.put(msg)

            # Periodically run background analysis every 30 frames (~30 seconds if interval=1)
            if self.frame_counter % 30 == 0:
                # Extract the last 30 frames for analysis
                frames_to_analyze = self.collected_frames[-30:]
                # Create a separate task for background analysis
                asyncio.create_task(self.run_background_analysis(frames_to_analyze))

            await asyncio.sleep(interval)

    @traceable
    async def run_background_analysis(self, frames: List[str]):
        """Run background analysis on collected frames."""
        logging.info("Running background analysis...")
        # Analyze last 30 frames for detailed insights
        if len(frames) >= 10:
            # Use a subset of frames for analysis to save processing
            recent_frames = frames[-10:]
            context = Context(
                game=self.global_context.game,
                category=self.global_context.category,
                focus_points=self.global_context.focus_points,
                notes=self.global_context.notes
            )
            analysis = await asyncio.to_thread(analyze_frame, len(self.collected_frames), recent_frames, context)
            # Update global context
            if analysis.get("new_notes"):
                self.global_context.notes.append(analysis["new_notes"])
            self.global_context.frame_analysis_results.append(analysis)
            logging.info(f"Frame analysis added for frames {analysis['timestamp_id']}.")

            # **New Code to Save Analysis to Data Folder**
            # Ensure the 'data' directory exists
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/analysis_reports", exist_ok=True)

            # Define the filename with timestamp to avoid overwriting
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = f"data/analysis_reports/frame_analysis_{timestamp}.json"

            # Save the analysis_result to the JSON file
            try:
                with open(analysis_filename, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logging.info(f"Frame analysis saved to '{analysis_filename}'.")
            except Exception as e:
                logging.error(f"Failed to save frame analysis: {e}")

    @traceable
    async def listen_audio(self):
        """Capture audio from the microphone and send to the model."""
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
        logging.info("Audio stream started.")

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
        """Send real-time media inputs to the model."""
        while True:
            msg = await self.out_queue.get()
            await self.ws.send(json.dumps(msg))

    @traceable
    async def receive_audio(self):
        """Receive audio responses from the model and play them."""
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
                logging.info("\n[Assistant] End of turn")

            tool_call = response.get('toolCall')
            if tool_call is not None:
                await handle_tool_call(self.ws, tool_call)

            server_content = response.get('serverContent')
            if server_content:
                self.handle_server_content(server_content)

    @traceable
    async def play_audio(self):
        """Play received audio responses."""
        pya_out = pyaudio.PyAudio()
        stream = pya_out.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True
        )
        logging.info("Audio playback started.")

        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    @traceable
    async def periodic_context_update(self, interval: int = 60):
        """Periodically update the assistant with the global context once per minute."""
        while True:
            await asyncio.sleep(interval)
            # Summarize the analysis results
            summary = await asyncio.to_thread(summarize_results, self.global_context.frame_analysis_results, Context(
                game=self.global_context.game,
                category=self.global_context.category,
                focus_points=self.global_context.focus_points,
                notes=self.global_context.notes
            ))
            # Optionally, you can send this summary to the assistant's memory or use it to influence responses
            logging.info("Periodic Context Update:")
            logging.info(summary)
            # Here, you can implement logic to update the assistant's knowledge based on the summary

    @traceable
    async def generate_final_report(self):
        """Generate and save the final report summarizing the gaming session."""
        logging.info("Generating final report...")
        summary = await asyncio.to_thread(summarize_results, self.global_context.frame_analysis_results, Context(
            game=self.global_context.game,
            category=self.global_context.category,
            focus_points=self.global_context.focus_points,
            notes=self.global_context.notes
        ))
        end_report = await asyncio.to_thread(generate_end_report, summary)
        logging.info("\n--- Final Summarized Report ---")
        print(end_report)
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/summary_reports", exist_ok=True)
        with open("data/summary_reports/end_report.txt", "w") as f:
            f.write(end_report)
        logging.info("Final report saved to 'data/end_report.txt'.")

    @traceable
    def handle_server_content(self, server_content):
        """Handle additional server content if needed."""
        model_turn = server_content.get('modelTurn')
        if model_turn:
            parts = model_turn.get('parts', [])
            for part in parts:
                executable_code = part.get('executableCode')
                if executable_code:
                    logging.info("-------------------------------")
                    logging.info("```python")
                    logging.info(executable_code.get('code', ''))
                    logging.info("```")
                    logging.info("-------------------------------")

                code_execution_result = part.get('codeExecutionResult')
                if code_execution_result:
                    logging.info("-------------------------------")
                    logging.info("```")
                    logging.info(code_execution_result.get('output', ''))
                    logging.info("```")
                    logging.info("-------------------------------")

        grounding_metadata = server_content.get('groundingMetadata')
        if grounding_metadata:
            # Handle grounding metadata if needed
            pass

    @traceable
    async def run_background_tasks(self, task_group: asyncio.TaskGroup):
        """Run background tasks such as periodic context updates."""
        task_group.create_task(self.periodic_context_update())

    @traceable
    async def run(self):
        """Run the agent by establishing WebSocket connection and starting tasks."""
        try:
            async with connect(URI, additional_headers={"Content-Type": "application/json"}) as ws, asyncio.TaskGroup() as tg:
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
                    },
                    {
                        "name": "perform_game_detection",
                        "description": "Detect the game and key focus points from provided frames.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "frames": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "Base64-encoded image frames."
                                    },
                                    "description": "List of base64-encoded image frames for game detection."
                                }
                            },
                            "required": ["frames"]
                        }
                    }
                ]
                await self.startup(tools=[{'function_declarations': tools_custom}])

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=10)

                # Start concurrent tasks
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.stream_screen_frames())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.send_text())
                tg.create_task(self.run_background_tasks(tg))

                # The 'async with' block will automatically wait for all tasks in the TaskGroup to finish
                # No need to call 'tg.wait_closed()'

        except asyncio.CancelledError:
            logging.info("Agent shutdown requested.")
        except Exception as e:
            logging.error("An error occurred:", exc_info=True)
            if self.audio_stream:
                self.audio_stream.close()

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    agent = Agent(global_context=global_context)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logging.info("Agent terminated by user.")
