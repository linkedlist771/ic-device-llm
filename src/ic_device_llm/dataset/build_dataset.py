from tqdm import tqdm
from typing import List, Union
from enum import Enum

from src.ic_device_llm.configs.path_config import RESOURCES_DIR_PATH
from src.ic_device_llm.prompts.dataset_prompt import UserPrompt, AssistantPrompt
from src.ic_device_llm.schemas import Conversation, Role, Message
import json

from src.ic_device_llm.utils.sampling_utils import get_sampler


class ConversationsBuilder:
    """这是对于单独一个群聊的信息"""

    def __init__(self, device_data_json_path: str):
        self.device_data = json.load(open(device_data_json_path, "r"))

    def build_conversations(self, *, sampleing_strategy: str) -> List[Conversation]:
        # TODO: 后面修改采样方法
        """
        构建对话的数据集
        :param qq: qq号
        :param depth: 对话的搜索长度（这里定死为1）
        :return: 对话列表
        """
        sampler = get_sampler(sampleing_strategy)
        # 我这里采用随机比例采样吧
        ratio = 0.005
        conversations = []

        for k, v in tqdm(
            self.device_data.items(), desc="Building conversations for each device"
        ):
            # "lg_nw": 10.0,
            # "r_nw": 2.0,
            # "t_ox(nm)": 0.75,
            # "index": 2,
            # "data": {
            lg_nw = v["lg_nw"]
            r_nw = v["r_nw"]
            t_ox_nm = v["t_ox(nm)"]
            # vg = v["vg"]
            data = v["data"]
            vg = data["vg"]
            sampled_data_idx = sampler(vg, ratio)

            __vg = [vg[i] for i in sampled_data_idx]
            __vd = [data["vd"][i] for i in sampled_data_idx]
            __ids = [data["ids"][i] for i in sampled_data_idx]

            for _vg, _vd, _ids in zip(__vg, __vd, __ids):
                user_prompt = UserPrompt(
                    lg_nw=lg_nw, r_nw=r_nw, t_ox_nm=t_ox_nm, vg=_vg, vd=_vd
                )
                user_prompt_str = user_prompt.render_prompt()
                assistant_prompt = AssistantPrompt(ids=_ids)
                assistant_prompt_str = assistant_prompt.render_prompt()
                conversation = Conversation(
                    messages=[
                        Message(role=Role.USER, content=user_prompt_str),
                        Message(role=Role.ASSISTANT, content=assistant_prompt_str),
                    ]
                )
                conversations.append(conversation)

        return conversations


class DataFormat(Enum):
    VICUNA = "vicuna"
    OPENAI = "openai"

    @staticmethod
    def convert_conversation_to_target_format(
        conversation_list: List[Conversation],
        data_format: "DataFormat",
        system_prompt: Union[None, str],
    ) -> List[dict]:
        if data_format == DataFormat.VICUNA:
            index = 0
            result = []
            for i in tqdm(conversation_list, total=len(conversation_list)):
                current_conversation = conversation_list[index]
                conversations = []
                for message in current_conversation.messages:
                    if message.role == Role.USER:
                        conversations.append(
                            {"from": Role.HUMAN.value, "value": message.content}
                        )
                    elif message.role == Role.ASSISTANT:
                        conversations.append(
                            {"from": Role.GPT.value, "value": message.content}
                        )
                _id = f"identity_{index}"
                result.append({"id": _id, "conversations": conversations})
                index += 1
            return result
        elif data_format == DataFormat.OPENAI:
            if isinstance(system_prompt, str):
                if system_prompt == "":
                    raise ValueError("system_prompt can not be empty")
            else:
                raise ValueError("system_prompt must be a string")
            index = 0
            result = []
            for i in tqdm(conversation_list, total=len(conversation_list)):
                current_conversation = conversation_list[index]
                conversations = [
                    {"role": Role.SYSTEM.value, "content": system_prompt}
                ]  # initialize the conversations with a system message
                for message in current_conversation.messages:
                    if message.role == Role.USER:
                        conversations.append(
                            {"role": Role.USER.value, "content": message.content}
                        )
                    elif message.role == Role.ASSISTANT:
                        conversations.append(
                            {"role": Role.ASSISTANT.value, "content": message.content}
                        )
                result.append({"messages": conversations})
                index += 1
            return result
        else:
            raise NotImplementedError("Not implemented data format")


def save_conversation_data(
    *,
    data_format: DataFormat,
    conversations: List[Conversation],
    save_path: str,
    system_prompt: Union[None, str] = None,
):
    if data_format == DataFormat.VICUNA:
        vicuna_format_data = DataFormat.convert_conversation_to_target_format(
            conversations, data_format
        )
        # save this list into the json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vicuna_format_data, f, ensure_ascii=False, indent=4)
    elif data_format == DataFormat.OPENAI:
        openai_format_data = DataFormat.convert_conversation_to_target_format(
            conversations, data_format, system_prompt
        )
        # save this list into the json
        assert str(save_path).endswith(".jsonl"), "save_path must be a jsonl file"
        with open(save_path, "w", encoding="utf-8") as f:
            for i in openai_format_data:
                json.dump(i, f, ensure_ascii=False)
                f.write("\n")

    else:
        raise NotImplementedError("Not implemented data format")


if __name__ == "__main__":
    processed_data_path = RESOURCES_DIR_PATH / "processed_data.json"
    conversations_builder = ConversationsBuilder(processed_data_path)
    conversations = conversations_builder.build_conversations(
        sampleing_strategy="ratio_uniform"
    )
    save_conversation_data(
        data_format=DataFormat.OPENAI,
        conversations=conversations,
        save_path=RESOURCES_DIR_PATH / "openai_conversations.jsonl",
        system_prompt="You are a helpful assistant on GAA devices.",
    )
