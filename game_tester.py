import time
import io
import base64
import pyautogui
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from typing import Literal, Optional
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

class Enemy(BaseModel):
    name: str
    health: int
    x_pos: int
    y_pos: int
class Card(BaseModel):
    name: str
    cost: int
    description: str
    type: str
    x_pos: int
    y_pos: int
class ScenaDescription(BaseModel):
    card_count: int
    card_list: list[Card]
    enemy_count: int
    enemy_list: list[Enemy]
    energy_left: int



class NextAction(BaseModel):
    """Possible moves for a turn-based card battle game."""
    click_type: Literal["none", "single", "swap", "drag"]
    x1: Optional[int]
    y1: Optional[int]
    x2: Optional[int]
    y2: Optional[int]
    explanation: str
class Gamer(BaseModel):
    situation: ScenaDescription
    action: NextAction

SYSTEM_PROMPT = """
You are an agent playing a turn-based card battle game similar to Slay the Spire.

Given a full-resolution screenshot of the game state, output exactly one of the following actions in JSON format:

  • no action (e.g. waiting or doing nothing):
    {"click_type": "none"}

  • single click (e.g. clicking "End Turn", "Next", "Reward"):    
    {"click_type": "single", "x1": <X>, "y1": <Y>}

  • swap (for match-type interactions):
    {"click_type": "swap", "x1": <X1>, "y1": <Y1>, "x2": <X2>, "y2": <Y2>}

  • drag (e.g. playing a card by click-and-drag to a target):
    {"click_type": "drag", "x1": <X1>, "y1": <Y1>, "x2": <X2>, "y2": <Y2>}

### Game mechanics:
- Cards are played by **clicking and dragging** them from the hand to the target (enemy or self).
- Some cards require a target (like attacks), some do not (like many skills or powers).
- Energy is limited per turn (shown top left); unavailable cards are visually dimmed.
- Enemies have "intents" above their heads that indicate their next action (e.g. attack, defend).
- End turn button is usually on the bottom right.
- Between fights, player clicks to select rewards, proceed through the map, or upgrade cards.

Coordinates must always be pixel-accurate positions **based on the original screenshot**, using the visible UI (e.g. card hand, enemies, buttons) to find the center points.
Use the grid overlay to help you find the coordinates. Grid size is 50x50 pixels, and the coordinates are showing the top left corner of the grid square. The coordinates are in the format (x, y) where x is the horizontal position and y is the vertical position.
Aim for the center of the target, and use the grid to find the pixel coordinates.

Please, double check that the coordinates are correctly pointing the object you are planning to click before outputting them. The coordinates grid is the main source of truth! If you are not sure, output "none" and wait for the next screenshot.

Always output a single valid JSON object. Do not output any text or explanation.

Game details: 
When the energy is 0, click on the next turn button.
On the map screen, click on the next node.
In the end of the turn choose the most powerful card.

"""

client = OpenAI()

def grab_and_prepare(grid_size=50, font_size=12, debug_save_path="screenshot_grid.png"):
    img = pyautogui.screenshot().convert("RGBA")
    w0, h0 = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # add the grid with coordinates on both sides
    for x in range(0, w0+1, grid_size):
        draw.line([(x,0),(x,h0)], fill=(255,255,255,128))
        draw.text((x+2,2), str(x), font=font, fill=(255,0,0,255))
        draw.text((x+2,h0-20), str(x), font=font, fill=(255,0,0,255))

    for y in range(0, h0+1, grid_size):
        draw.line([(0,y),(w0,y)], fill=(255,255,255,128))
        # at top
        draw.text((2,y+2), str(y), font=font, fill=(255,0,0,255))
        # at bottom
        draw.text((w0-20,y+2), str(y), font=font, fill=(255,0,0,255))

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    combined.save(debug_save_path, format="PNG")
    return b64, (w0, h0)

while True:
    b64, (w0, h0) = grab_and_prepare()
    data_url = f"data:image/png;base64,{b64}"

    resp = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text",  "text": f"Board screenshot: {w0}x{h0}."},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=Gamer,
    )
    print("Situation: ", resp.output_parsed.situation)
    action = resp.output_parsed.action
    print("GPT ➜", action)

    if action.click_type == "single" and action.x1 is not None:
        pyautogui.click(x=action.x1, y=action.y1, button="left")

    elif action.click_type == "swap" and action.x1 is not None and action.x2 is not None:
        pyautogui.click(x=action.x1, y=action.y1, button="left")
        time.sleep(0.1)
        pyautogui.click(x=action.x2, y=action.y2, button="left")

    elif action.click_type == "drag" and action.x1 is not None and action.x2 is not None:
        pyautogui.mouseDown(x=action.x1, y=action.y1, button="left")
        time.sleep(0.1)
        pyautogui.moveTo(x=action.x2, y=action.y2, duration=0.5)
        time.sleep(0.1)
        pyautogui.mouseUp(button="left")

