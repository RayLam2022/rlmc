# ref: https://blog.csdn.net/LuohenYJ/article/details/132405814
# demo, not used in the project
import pyaudio

RECORD_SECONDS = 5
CHUNK = 1024
RATE = 16000

p = pyaudio.PyAudio()
# frames_per_buffer设置音频每个缓冲区的大小
stream = p.open(
    format=p.get_format_from_width(2),
    channels=1,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)

print("recording")
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # read读取音频然后writer播放音频
    stream.write(stream.read(CHUNK))
print("done")

stream.close()
p.terminate()
