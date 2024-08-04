# ref: https://juejin.cn/post/7374055681890795555

from typing import Callable, Any
import random
import time


__all__ = ["user_login"]  # 装饰器

users_db = {"user-a": "123password", "user-b": "456password", "user-c": "789password"}

# 该字典用于存放各个用户登录密码错误的次数
login_counter = {}


def generate_captcha() -> str:
    """生成四位数字的验证码"""
    return str(random.randint(1000, 9999))


def verify_captcha(captcha_input: str, captcha_actual: str) -> bool:
    """验证用户输入的验证码是否正确"""
    return captcha_input == captcha_actual


def user_login(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        while True:
            """输入用户登录信息"""
            username = input("请输入用户名：")
            password = input("请输入密码：")

            # 生成验证码
            captcha = generate_captcha()
            print(f"验证码是：{captcha}")
            captcha_input = input("请输入验证码：")

            if username in users_db:

                if not verify_captcha(captcha_input, captcha):
                    print("验证码错误！")
                    continue
                # 密码验证
                if users_db[username] == password:
                    print("登录成功！")
                    return func(*args, **kwargs)
                    break  # 登录成功，退出循环
                else:  # 密码错误次数记录与锁定账号
                    if username in login_counter:
                        login_counter[username] += 1
                    else:
                        login_counter[username] = 1

                    if login_counter[username] >= 3:
                        print("密码错误次数过多，账号已被锁定！")
                        break  # 输错三次密码，锁定账号，退出循环
                    else:
                        print("密码错误！请重新输入！")
            else:
                print("用户名不存在，请重新输入！")

    return wrapper


if __name__ == "__main__":

    @user_login
    def funcion(x, y):
        return x + y

    print(funcion(6, 9))
