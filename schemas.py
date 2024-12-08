from pydantic import BaseModel, Field
from typing import List

class FrameAnalysis(BaseModel):
    comments: str = Field(..., description="General commentary on the frame.")
    recommendations: str = Field(..., description="Suggestions for improvement.")
    tricks_used: str = Field(..., description="Tricks or optimizations observed.")
    good_actions: str = Field(..., description="Actions done well.")
    bad_actions: str = Field(..., description="Mistakes or inefficiencies detected.")
    new_notes: str = Field(..., description="New global information, that shall be added to the context of the whole video, because it is relevant for the whole video and important to the overall video understanding. Better to leave it empty if there is no new globally important information to add.")


class Context(BaseModel):
    game: str
    category: str
    notes:  List[str]