import argparse
import timeit
from importlib import import_module


parser = argparse.ArgumentParser("cuda_test")
parser.add_argument("-m", "--module", required=True, help="the module to test")
args = parser.parse_args()


__all__ = ["cudatest"]


def torchtest(th) -> None:
    print(th.__version__)
    tensor = th.Tensor([1.0])
    tensor = tensor.cuda()
    print(th.backends.cudnn.is_acceptable(tensor))


def tftest(tf) -> None:

    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_cuda())

    with tf.device("/cpu:0"):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        print(cpu_a.device, cpu_b.device)

    with tf.device("/gpu:0"):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        print(gpu_a.device, gpu_b.device)

    def cpu_run():
        with tf.device("/cpu:0"):
            met = tf.matmul(cpu_a, cpu_b)
        return met

    def gpu_run():
        with tf.device("/gpu:0"):
            met = tf.matmul(gpu_a, gpu_b)
        return met

    # warm up
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print("warmup:", cpu_time, gpu_time)

    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print("run time:", cpu_time, gpu_time)


def cudatest() -> None:
    assert args.module in ["torch", "tensorflow"], "module参数错误,可选torch,tensorflow"
    if args.module == "torch":
        try:
            th = import_module("torch")
            tv = import_module("torchvision")
            print("torchvision version", tv.__version__)
            torchtest(th)
        except:
            print("torch或torchvision未安装或者存在安装异常")
    elif args.module == "tensorflow":
        try:
            tf = import_module("tensorflow")
            tftest(tf)
        except:
            print("tensorflow未安装,超过测试版本或者存在安装异常")


if __name__ == "__main__":
    cudatest()
