# 注意余额或key变更，插件调用，在百度智能云更改插件，“插件编排”，上线下线，在插件编辑中重新上传知识库文件。目前问题是章节不会区分，基本只会按整个文件进行理解搜索，部分理解可能也不太对，需要对结果校对
# https://demo.ragflow.io/
import argparse
import os
import pyttsx3

import qianfan

parser = argparse.ArgumentParser("qianfan")
parser.add_argument(
    "-a",
    "--access_key",
    required=True,
)
parser.add_argument(
    "-s",
    "--secret_key",
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    default="ERNIE-Bot-turbo",  # ERNIE-4.0-8K ChatLaw
    help="ERNIE-4.0-8K, ChatLaw, ERNIE-Bot-turbo...",
)

parser.add_argument(
    "-sp",
    "--is_speak",
    action="store_true",
)

args = parser.parse_args()


# 【推荐】使用安全认证AK/SK鉴权，通过环境变量初始化认证信息
# 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk
os.environ["QIANFAN_ACCESS_KEY"] = args.access_key
os.environ["QIANFAN_SECRET_KEY"] = args.secret_key


def main() -> None:
    chat_comp = qianfan.ChatCompletion()

    msgs = qianfan.Messages()
    while True:
        msgs.append(input("master:"))
        resp = chat_comp.do(model=args.model, messages=msgs)
        print("assistant:", resp["result"])
        if args.is_speak:
            pyttsx3.speak(resp["result"])
        msgs.append(resp)


if __name__ == "__main__":
    main()
