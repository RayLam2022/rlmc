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
        attributes: AgentAttributes,
        function: Callable,
        role: Literal["user", "assistant", "function_call"] = "assistant",
    ) -> None:

        self.attributes = attributes
        self.agent_name = self.attributes.name
        self.func = function
        self.role = role

    def gen(
        self,
    ) -> Generator[
        Tuple[str, Response, List[Response]],
        Tuple[Dict[str, "Agent"], int, List[Response]],
        None,
    ]:
        active_agent_name = None
        response_dict = None
        history = None
        while True:
            agents_dict, response_id, history = (
                yield active_agent_name,
                response_dict,
                history,
            )
            response_dict = dict()
            response_dict["response_id"] = response_id
            response_dict["agent_name"] = self.agent_name
            response_dict["agent_role"] = self.role

            # agent生成内容
            resp = f"{self.agent_name}: xbb"
            time.sleep(0.5) 

            response_dict["response"] = resp
            response_dict = Response(**response_dict)
            logger.info(f"{resp}")

            active_agent_name = self._switch_agent(agents_dict)
            logger.info(f"choice:{active_agent_name}")

            history.append(response_dict)

    def _switch_agent(self, agents_dict: Dict[str, "Agent"]) -> str:
        """_summary_

        Args:
            agents_dict (Dict[str,Agent]): _description_

        Returns:
            str: active_agent_name
        """
        active_agent_name = random.choice(list(agents_dict.keys()))
        return active_agent_name


class Hornet:
    def __init__(
        self,
        agents: List[Agent],
        max_turns: int = float("inf"),
    ) -> None:
        self.agents_dict = dict()
        for agent in agents:
            generator = agent.gen()
            generator.send(None)
            self.agents_dict[agent.agent_name] = generator

        self.max_turns = max_turns
        self.history = []

    def run(self, start_agent_name: str) -> Tuple[Response, List[Response]]:
        counter = 0
        active_agent_name = start_agent_name
        while counter < self.max_turns:
            logger.debug(counter)
            active_agent = self.agents_dict[active_agent_name]
            response_id = counter
            active_agent_name, response_dict, self.history = active_agent.send(
                (self.agents_dict, response_id, self.history)
            )

            counter += 1
        last_response_dict = response_dict
        return last_response_dict, self.history


def add(a, b):
    return a + b


def multi(a, b):
    return a * b


def sub(a, b):
    return a - b


def add2(a, b):
    return a + b + 1


if __name__ == "__main__":
    try:
        address = Address(
            nationality="American", street="123", city="New York", zip_code="10001"
        )
        attributes = {
            "username": "A",
            "gender": "male",
            "birthday": datetime.datetime.strptime("2000-11-11", "%Y-%m-%d"),
            "health_status": "healthy",
            "state": "alive",
            "character": "friendly",
            "hobby": "reading",
            "height": 180,
            "weight": 75,
            "education_level": "bachelor",
            "skill": "programming",
            "staff": "engineer",
            "address": address,
            "language": "English",
            "instruction": "Please help me find the book I lost.",
            "target": "book",
            "e_mail": "john@example.com",
        }
        attrib = AgentAttributes(**attributes)
    except ValidationError as e:
        rprint(f"[red]数据验证错误：{e.errors()}")
        sys.exit(1)

    agent_a = Agent(
        attributes=attrib,
        function=add,
    )
    attributes["username"] = "B"
    attrib_b = AgentAttributes(**attributes)
    agent_b = Agent(
        attributes=attrib_b,
        function=multi,
    )
    attributes["username"] = "C"
    attrib_c = AgentAttributes(**attributes)
    agent_c = Agent(
        attributes=attrib_c,
        function=sub,
    )

    hornet = Hornet([agent_a, agent_b, agent_c], max_turns=6)
    last_resp, history = hornet.run("A")
    rprint(f"[green]Results:{history}")
    rprint(f"[green]Final result:{last_resp}")
