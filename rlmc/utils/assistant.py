'''
@File    :   assistant.py
@Time    :   2024/07/04 12:21:43
@Author  :   RayLam
@Contact :   1027196450@qq.com
'''


import pyaudio

RECORD_SECONDS = 5
CHUNK = 4096
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(2),
    channels=1,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)

print("Start Recording")
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    stream.write(stream.read(CHUNK))

stream.close()
p.terminate()

print("Done")
