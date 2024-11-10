"""
@File    :   agents.py
@Time    :   2024/11/10 10:16:14
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

##目前仅支持无分支单向图
import asyncio
import multiprocessing as mp
import threading as th
import sys
import os
import random
import time
import datetime
import uuid
from typing import Any, Optional, Union, Literal, Callable, Dict, List, Tuple


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
    "Regulator",
    "Workflow",
    "Agent",
    "AgentAttributes",
    "Address",
    "EdgeAttributes",
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


class EdgeAttributes(BaseModel):
    edge_name: str
    node_out_result: Any


class Agent(mp.Process):
    def __init__(
        self,
        mp_signal,
        mp_lock,
        send_queue,
        rec_queue,
        mgr_dict,
        result_dict,
        attributes: AgentAttributes,
        function: Callable,
    ):
        super().__init__(daemon=True)
        self.attributes = attributes
        self.agent_name = self.attributes.name
        self.func = function

        self.mp_signal = mp_signal
        self.mp_lock = mp_lock
        self.send_queue = send_queue
        self.rec_queue = rec_queue
        self.mgr_dict = mgr_dict

        self.result_dict = result_dict

    def run(self):
        while self.mp_signal.value:

            if (
                self.mgr_dict["task_launcher"] == self.agent_name
                and self.mgr_dict["res_launcher"] == None
                and self.mgr_dict["is_done"] == False
            ):
                self.mp_lock.acquire()
                gen_content = random.randint(0, 10)
                info = f"agent {self.agent_name} is running  {gen_content}"
                self.mgr_dict["res_launcher"] = gen_content
                # logger.info(info)
                self.mp_lock.release()
            elif (
                self.mgr_dict["task_receiver"] == self.agent_name
                and self.mgr_dict["res_launcher"] != None
                and self.mgr_dict["is_done"] == False
            ):
                self.mp_lock.acquire()
                receive = self.mgr_dict["res_launcher"]
                gen_content = random.randint(0, 10)
                res = self.func(receive, gen_content)
                info = f"agent {self.agent_name} is running reciever {receive} {gen_content} {res}"
                # logger.info(info)

                self.result_dict[self.mgr_dict["task_id"]] = (
                    self.mgr_dict["task_launcher"],
                    self.mgr_dict["task_receiver"],
                    receive,
                    res,
                    True,
                )
                self.mgr_dict["res_receiver"] = res
                self.mgr_dict["is_done"] = True
                self.mp_lock.release()

            else:
                task = random.randint(0, 10)
                info = f"agent {self.agent_name} is running  {task}"
                # logger.info(info)
            rprint(f"[green]{info}")
            time.sleep(1)


class Regulator:
    def __init__(
        self,
        workflow: nx.DiGraph,
        mp_signal,
        mp_lock,
        send_queue,
        rec_queue,
        mgr_dict,
        result_dict,
        agents: List[Agent],
    ) -> None:
        self.workflow = workflow
        self.mp_signal = mp_signal
        self.mp_lock = mp_lock
        self.send_queue = send_queue
        self.rec_queue = rec_queue
        self.mgr_dict = mgr_dict

        self.result_dict = result_dict  # {task_id: (task_launcher, task_receiver, res_launcher, res_receiver, is_done}
        self.agents_map: Dict[str, Agent] = {
            agent.attributes.name: agent for agent in agents
        }
        self.send_queue = send_queue
        self.rec_queue = rec_queue

    def run(self):

        for agent in self.agents_map.values():
            agent.start()

        # while True:
        #     if input("终止请按q\n") == "q":
        #         for agent in self.agents_map.values():
        #             agent.terminate()
        #         break

        node_result = None
        task_counter = 0
        for edge in self.workflow.edges:
            collect_dict, node_result = self._create_task(
                task_counter, edge, node_result
            )
            # print(collect_dict)
            task_counter += 1
            # break

        self.mp_signal.value = 0

        final_result = node_result

        return self.result_dict, final_result

    def _init_mgrdict(self) -> None:
        self.mgr_dict["task_id"] = None
        self.mgr_dict["task_receiver"] = None
        self.mgr_dict["task_launcher"] = None
        self.mgr_dict["is_done"] = False
        self.mgr_dict["res_launcher"] = None
        self.mgr_dict["res_receiver"] = None

    def _create_task(
        self, task_id: int, edge: Tuple[str, str], node_result: Any
    ) -> None:
        collect_dict = dict()
        self._init_mgrdict()
        self.mgr_dict["task_id"] = task_id

        launcher = self.agents_map[edge[0]]
        receiver = self.agents_map[edge[1]]

        if node_result is not None:
            self.mgr_dict["res_launcher"] = node_result
        self.mgr_dict["task_launcher"] = launcher.agent_name
        self.mgr_dict["task_receiver"] = receiver.agent_name

        while not self.mgr_dict["is_done"]:
            pass

        collect_dict[task_id] = (
            self.mgr_dict["task_launcher"],
            self.mgr_dict["task_receiver"],
            self.mgr_dict["res_launcher"],
            self.mgr_dict["res_receiver"],
            self.mgr_dict["is_done"],
        )
        node_result = self.mgr_dict["res_receiver"]
        self._init_mgrdict()
        return collect_dict, node_result


class Workflow:
    def __init__(
        self, method: Literal["bfs", "dfs"] = "bfs", is_show=False, is_save_graph=True
    ):
        G = nx.DiGraph()

        # 添加节点
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")

        # 添加边
        # G.add_edge("A", "B")
        # G.add_edge("C", "D")
        # G.add_edge("B", "D")
        # G.add_edge("A", "C")

        G.add_edge("A", "B", name="A_B")
        G.add_edge("A", "C", name="A_C")
        G.add_edge("C", "D", name="C_D")
        G.add_edge("B", "C", name="B_C")

        print(f'入邻居节点：{list(G.predecessors("C"))}')
        print(f'出邻居节点：{list(nx.neighbors(G, "A"))}')
        print('edge_name:',G.edges['A', 'B']['name'])

        # 深度优先遍历
        def dfs_traversal(G):
            for node in nx.dfs_preorder_nodes(G, 1):
                print(node)

        # 广度优先遍历
        def bfs_traversal(G):
            for node in nx.bfs_tree(G, 1):
                print(node)

        # 调用函数进行遍历
        # print("Depth-First Search:")
        # dfs_traversal(G)
        # print("\nBreadth-First Search:")
        # bfs_traversal(G)

        self.traversal = bfs_traversal if method == "bfs" else dfs_traversal
        self.graph = G

        self.in_degrees = G.in_degree()
        self.out_degrees = G.out_degree()
        self.edges = G.edges

        # 遍历图并打印节点及其入度
        # for node, in_degree in self.in_degrees:
        #     rprint(f"[blue]节点: {node}, 入度: {in_degree}")
        print(self.out_degrees)
        for node, out_degree in self.out_degrees:
            rprint(f"[green]节点: {node}, 出度: {out_degree}")
        print(self.edges)
        for edge in self.edges:
            rprint(f"[cyan]边: {edge}")

        nx.draw(
            G,
            with_labels=True,
            node_color="skyblue",
            edge_color="black",
            node_size=800,
            font_size=20,
            arrowstyle="->",
        )

        if is_save_graph:
            plt.savefig("./workflow.png")
        if is_show:
            plt.show()
        plt.clf()


def add(a, b):
    return a + b


def multi(a, b):
    return a * b


def sub(a, b):
    return a - b


def add2(a, b):
    return a + b + 1


if __name__ == "__main__":
    rprint(f"[green]cpu count:{mp.cpu_count()}")

    workflow = Workflow()

    # mp_signal = mp.Value("i", 1)
    # mp_lock = mp.Lock()
    # result_dict = mp.Manager().dict()
    # mgr_dict = mp.Manager().dict()
    # send_queue = mp.Queue()
    # rec_queue = mp.Queue()

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
