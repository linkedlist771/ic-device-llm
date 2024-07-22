from dataclasses import dataclass
from enum import Enum
from typing import List


class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    GPT = "gpt"
    HUMAN = "human"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class Conversation:
    messages: List[Message]
