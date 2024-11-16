"""
@File    :   agents_coroutine.py
@Time    :   2024/11/15 10:16:14
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import asyncio
import sys
import os
import random
import time
import datetime
import uuid
from typing import Any, Optional, Union, Literal, Callable, Dict, List, Tuple, Generator


from pydantic import BaseModel, Field, ValidationError
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
import logging
import numpy as np
import rich
import dotenv

dotenv.load_dotenv()
rprint = Console().print
rrule = Console().rule


__all__ = [
    "Agent",
    "AgentAttributes",
    "Address",
]


pid = os.getpid()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    f"%(asctime)s - [PID: {pid}] - %(name)s - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
file_handler = logging.FileHandler("./agents.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Address(BaseModel):
    nationality: str
    street: str
    city: str
    zip_code: str


class AgentAttributes(BaseModel):
    id: Union[uuid.UUID, str, int] = Field(default_factory=uuid.uuid1)
    name: str = Field(max_length=20, alias="username")
    gender: Literal["male", "female"]
    birthday: datetime.datetime
    health_status: Literal["healthy", "sick"]
    state: str
    character: str
    hobby: str
    height: int = Field(ge=0, le=300)  # cm
    weight: float = Field(ge=0.0, le=650.0)  # kg
    education_level: str
    skill: str
    staff: str
    address: Address
    language: str
    e_mail: Optional[str] = None

    instruction: str
    target: str

    @property
    def age(self) -> int:
        birth_date = self.birthday
        today = datetime.date.today()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age


class Response(dict):
    def __init__(
        self,
        response_id: int,
        agent_name: str,
        agent_role: Literal["user", "assistant", "function_call"],
        response: str,
    ):
        super().__init__(
            response_id=response_id,
            agent_name=agent_name,
            agent_role=agent_role,
            response=response,
        )


class Agent:
    def __init__(
        self,
        agents: List["Agent"],
        history: List[Response],
        response_dict:Response,
        attributes: AgentAttributes,
        function: Callable,
        max_turns:int=float("inf"),
    ):
        self.agents = agents
        self.attributes = attributes
        self.agent_name = self.attributes.name
        self.func = function

        self.history = history
        self.response_dict = response_dict
        self.max_turns = max_turns

    def run(self):
        while True:
            info = ""
            rprint(f"[green]{info}")
            time.sleep(0.5)

    def _handle_agent(self, response: Response):
        self.history.append(response)

    def _handle_history(self, response_dict: Response):
        self.history.append(response_dict)


def add(a, b):
    return a + b


def multi(a, b):
    return a * b


def sub(a, b):
    return a - b


def add2(a, b):
    return a + b + 1


if __name__ == "__main__":
    a = Response(response_id=1, agent_name="a", response="ss")
    print(a)

    # try:
    #     address = Address(
    #         nationality="American", street="123", city="New York", zip_code="10001"
    #     )
    #     attributes = {
    #         "username": "A",
    #         "gender": "male",
    #         "birthday": datetime.datetime.strptime("2000-11-11", "%Y-%m-%d"),
    #         "health_status": "healthy",
    #         "state": "alive",
    #         "character": "friendly",
    #         "hobby": "reading",
    #         "height": 180,
    #         "weight": 75,
    #         "education_level": "bachelor",
    #         "skill": "programming",
    #         "staff": "engineer",
    #         "address": address,
    #         "language": "English",
    #         "instruction": "Please help me find the book I lost.",
    #         "target": "book",
    #         "e_mail": "john@example.com",
    #     }
    #     attrib = AgentAttributes(**attributes)
    # except ValidationError as e:
    #     rprint(f"[red]数据验证错误：{e.errors()}")
    #     sys.exit(1)

    # agent_a = Agent(
    #     mp_signal=mp_signal,
    #     mp_lock=mp_lock,
    #     send_queue=send_queue,
    #     rec_queue=rec_queue,
    #     mgr_dict=mgr_dict,
    #     result_dict=result_dict,
    #     attributes=attrib,
    #     function=add,
    # )
    # attributes["username"] = "B"
    # attrib = AgentAttributes(**attributes)
    # agent_b = Agent(
    #     mp_signal=mp_signal,
    #     mp_lock=mp_lock,
    #     send_queue=send_queue,
    #     rec_queue=rec_queue,
    #     mgr_dict=mgr_dict,
    #     result_dict=result_dict,
    #     attributes=attrib,
    #     function=multi,
    # )
    # attributes["username"] = "C"
    # attrib = AgentAttributes(**attributes)
    # agent_c = Agent(
    #     mp_signal=mp_signal,
    #     mp_lock=mp_lock,
    #     send_queue=send_queue,
    #     rec_queue=rec_queue,
    #     mgr_dict=mgr_dict,
    #     result_dict=result_dict,
    #     attributes=attrib,
    #     function=sub,
    # )
    # attributes["username"] = "D"
    # attrib = AgentAttributes(**attributes)
    # agent_d = Agent(
    #     mp_signal=mp_signal,
    #     mp_lock=mp_lock,
    #     send_queue=send_queue,
    #     rec_queue=rec_queue,
    #     mgr_dict=mgr_dict,
    #     result_dict=result_dict,
    #     attributes=attrib,
    #     function=add2,
    # )

    # regulator = Regulator(
    #     workflow=workflow,
    #     mp_signal=mp_signal,
    #     mp_lock=mp_lock,
    #     send_queue=send_queue,
    #     rec_queue=rec_queue,
    #     mgr_dict=mgr_dict,
    #     result_dict=result_dict,
    #     agents=[agent_a, agent_b, agent_c, agent_d],
    # )
    # results, final_result = regulator.run()
    # rprint(f"[green]Results:{results}")
    # rprint(f"[green]Final result:{final_result}")
