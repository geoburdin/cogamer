import os
from video_utils import extract_frames
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from schemas import FrameAnalysis, Context
import dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable

dotenv.load_dotenv()

# --- Configuration ---
FRAME_RATE = 1  # Extract 1 frame per second
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain Model Setup
# model = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)  # Uncomment for OpenAI
structured_llm = model.with_structured_output(FrameAnalysis)

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
             Your goal is to analyze this segment of the video, identify key strategies, tricks, and inefficiencies, and contribute to the global understanding of the speedrun.

             Focus on the following:
             1. **Frame-Specific Analysis**:
                 - Identify tricks, skips, glitches, or optimizations used.
                 - Provide professional-level commentary with precise terminology (e.g., wall clips, route skips, jump chaining, frame-perfect tricks).
                 - Highlight good execution, mistakes, and inefficiencies.

             2. **Global Notes** (New Knowledge for the Speedrun):
                 - Add only information that improves the global understanding of the entire speedrun.
                 - For example: overall routing decisions, consistent techniques being used, recurring glitches, or notable strategies seen in this segment, storylines, or character development.
                 - Leave this field empty if no new global information is detected.

             Current Global Context:
             {context.notes}

             Here are the frames from the last 10 seconds (Frame {frame_id-10} to {frame_id}):
             """},
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    # Invoke the structured model
    result = structured_llm.invoke([message])
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

# placeholder Summarization ---
@traceable
def summarize_results(results: List[dict], context: Context) -> str:
    """Summarize the results from all frames."""
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
    Here are the analyses from per second frames extracted from a video.

    {summary}

    Please analyze these findings and prepare a detailed speedrun report. Highlight:
    - Describe the overall gameplay, feel free to write a narrative
    - Overall performance
    - Common mistakes or inefficiencies
    - Best tricks observed
    - Key recommendations for improvement
    """

    # Use the model to generate the end report
    report_message = HumanMessage(content=[{"type": "text", "text": report_request}])
    result = model.invoke([report_message])

    # Return the textual report
    return result.content


@traceable
def video_report(video_path: str):
    # Step 1: Extract frames
    print("Extracting frames from video...")
    frames = extract_frames(video_path, FRAME_RATE)
    print(f"Extracted {len(frames)} frames.")

    # Step 2: Initialize context and process frames
    results = []
    global_context = Context(game="Minecraft", category="Speedrun Analysis", notes=[])

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