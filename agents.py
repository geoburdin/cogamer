import os
import random
from video_utils import extract_frames
from typing import List
from langchain_openai import ChatOpenAI
from schemas import FrameAnalysis, Context, DetectGameFocusPoints
import dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

dotenv.load_dotenv()

# --- Configuration ---
FRAME_RATE = 1  # Extract 1 frame per second
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain Model Setup
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
structured_llm_frame_analysis = model.with_structured_output(FrameAnalysis)
structured_llm_detect_game_focus_points = model.with_structured_output(DetectGameFocusPoints)

# --- Sub-Agent for Game Detection and Focus Points ---
@traceable
def detect_game_and_focus_points(frames_data: List[str]) -> dict:
    """Analyze random frames to detect the game and focus points for analysis."""
    print("Sub-Agent: Analyzing random frames to detect game and focus points...")

    message = HumanMessage(
        content=[
                    {"type": "text",
                     "text": """
             You are an expert gaming analyst. Analyze the following gameplay frames to detect:
             1. The name of the game (if possible).
             2. Key focus points to analyze during the video, such as specific tricks, objects important for the gamer community or strategies.
             Be specific, professional, and use gaming terminology."""},
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    # Invoke the model
    result = structured_llm_detect_game_focus_points.invoke([message])
    result = result.model_dump()
    print("Sub-Agent Results:", result)

    # Parse the content into a usable dictionary
    return result

# --- Frame Analysis Function ---
@traceable
def analyze_frame(frame_id: int, frames_data: List[str], context: Context) -> dict:
    """Analyze a batch of frames and return the structured output."""
    print(f"Analyzing frame {frame_id}...")
    message = HumanMessage(
        content=[
                    {"type": "text",
                     "text": f"""
             You are a professional speedrun commentator analyzing a gameplay video of '{context.game}' in the category '{context.category}'.
             Focus on these key points during analysis: {context.focus_points}.
             
             Provide the following:
             1. Frame-specific analysis:
                 - Tricks, skips, glitches, and movement optimizations.
                 - Mistakes or inefficiencies.
                 - Execution precision and routing decisions.
             2. New global notes for the speedrun (if applicable)."""},
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    # Invoke the structured model
    result = structured_llm_frame_analysis.invoke([message])
    result = result.model_dump()

    result["timestamp_id"] = f"{frame_id-10}-{frame_id} seconds of the video"
    print(f"Analysis Result: {result}")

    # Update global context if new global notes are provided
    if result.get("new_notes"):
        print(f"Updating global context with new note: {result['new_notes']}")
        context.notes.append(result["new_notes"])

    # Save individual frame analysis
    with open(f"data/frame_{frame_id}_analysis.json", "w") as f:
        f.write(str(result))

    return result

# --- Summarization Function ---
@traceable
def summarize_results(results: List[dict], context: Context) -> str:
    summary = {
        "total_frames_analyzed": len(results),
        "global_context": context.notes,
        "comments": [res["comments"] for res in results],
        "recommendations": [res["recommendations"] for res in results],
        "tricks_used": [res["tricks_used"] for res in results],
        "good_actions": [res["good_actions"] for res in results],
        "bad_actions": [res["bad_actions"] for res in results]
    }

    return str(summary)

# --- Generate End Report ---
@traceable
def generate_end_report(summary: str) -> str:
    """Generate an end report summarizing the entire speedrun."""
    print("\nGenerating the end report...")

    # Prepare text content for the model
    report_request = f"""
    Here is the structured analysis of a gameplay video:

    {summary}

    Write a detailed report highlighting:
    - The overall gameplay and performance
    - Key mistakes or inefficiencies
    - Best strategies, tricks, and optimizations
    - Final recommendations for improvement.
    """

    # Use the model to generate the end report
    report_message = HumanMessage(content=[{"type": "text", "text": report_request}])
    result = model.invoke([report_message])

    # Return the textual report
    return result.content

# --- Main Video Analysis Function ---
@traceable
def video_report(video_path: str):
    # Step 1: Extract frames
    print("Extracting frames from video...")
    frames = extract_frames(video_path, FRAME_RATE)
    print(f"Extracted {len(frames)} frames.")

    # Step 1: Sub-Agent to analyze random frames
    random_indices = random.sample(range(len(frames)), min(10, len(frames)))
    random_frames = [frames[i] for i in random_indices]
    initial_analysis = detect_game_and_focus_points(random_frames)
    print(f"Sub-Agent Findings: {initial_analysis}")

    # Initialize global context
    global_context = Context(
        game=initial_analysis.get("game", "Unknown"),
        category="Speedrun Analysis",
        focus_points=initial_analysis.get("focus_points", []),
        notes=[]
    )

    # Step 2: Analyze all frames
    results = []
    for i in range(10, len(frames), 10):  # Sliding window of 10 frames
        batch_frames = frames[i-10:i]
        analysis = analyze_frame(i, batch_frames, global_context)
        results.append(analysis)

    # Step 3: Summarize results
    print("\n--- Generating Final Summary ---")
    summary = summarize_results(results, global_context)
    print(summary)

    with open("data/summary.json", "w") as f:
        f.write(summary)

    # Step 4: Generate End Report
    end_report = generate_end_report(summary)
    print("\n--- End Report ---")
    print(end_report)

    with open("data/end_report.txt", "w") as f:
        f.write(end_report)

    print("\n--- Video Analysis Completed ---")
    return end_report
