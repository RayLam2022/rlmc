# ref: https://blog.csdn.net/Listest/article/details/121157975?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-121157975-blog-84341801.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-121157975-blog-84341801.235%5Ev43%5Epc_blog_bottom_relevance_base8&utm_relevant_index=2
# demo, not used in the project
import wave
import threading
from os import remove,mkdir,listdir
from os.path import exists,splitext,basename,join
from datetime import datetime
from time import sleep
import pyaudio
from PIL import ImageGrab
from numpy import array
import cv2
from moviepy.editor import *
 

MODE='camera'   # 'screen' or 'camera'
CHUNK_sIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 48000
allowRecording = True

 
def record_audio():
    p= pyaudio.PyAudio()
    # event.wait()
    sleep(3)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    # input_device_index=4,#立体混音，具体选哪个根据需要选择
                    output=True,
                    frames_per_buffer = CHUNK_sIZE)
    wf = wave.open(audio_filename,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    while allowRecording:
        # 从录音设备读取数据，直接写入wav文件
        data = stream.read(CHUNK_sIZE)
        wf.writeframes(data)
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
 
def record_cam():
    # 录制屏幕
    cap=cv2.VideoCapture(0)  # camera id
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps=cap.get(cv2.CAP_PROP_FPS)
    video =cv2.VideoWriter(screen_video_filename,
                           cv2.VideoWriter_fourcc(*'XVID'),
                           fps,(frame_width,frame_height)) #帧速和视频宽度、高度
    while cap.isOpened() and allowRecording:
        ret,im=cap.read()
        if ret:
            video.write(im)
        else:
            break
    cap.release()
    video.release()


def record_screen():
    # 录制屏幕
    im = ImageGrab.grab()
    video =cv2.VideoWriter(screen_video_filename,
                           cv2.VideoWriter_fourcc(*'XVID'),
                           25,im.size) #帧速和视频宽度、高度
    while allowRecording:
        im = ImageGrab.grab()
        im = cv2.cvtColor(array(im),cv2.COLOR_RGB2BGR)
        video.write(im)
    video.release()
 
 
root=r"C:\Users\RayLam\Desktop\temp_file"
now = str(datetime.now())[:19].replace(':','_')
audio_filename = join(root,"%s.mp3"%now)
webcam_video_filename = join(root,"t%s.avi"%now)
screen_video_filename = join(root,"tt%s.avi"%now)
video_filename =join(root, "%s.avi"%now)
 
#创建两个线程，分别录音和录屏
t1 = threading.Thread(target=record_audio)
if MODE=='screen':
    t2 = threading.Thread(target=record_screen)
else:
    t2 = threading.Thread(target=record_cam)
 
 
event = threading.Event()
event.clear()
for t in (t1,t2):
    t.start()
# 等待摄像头准保好，提示用户三秒钟以后开始录制
# event.wait()
print('3秒后开始录制，按q键回车结束录制')
while True:
    if input() =='q':
        break
allowRecording = False
for i in (t1,t2):
    t.join()
 
#把录制的视频和音频合成视频文件
audio = AudioFileClip(audio_filename)
video1 = VideoFileClip(screen_video_filename)
ratio1 = audio.duration / video1.duration
video1 = (video1.fl_time(lambda t: t/ratio1,apply_to=['video'])\
            .set_end(audio.duration))
            
video = CompositeVideoClip([video1]).set_audio(audio)
video.write_videofile(video_filename,codec= 'libx264',fps = 25)
 
remove(audio_filename)
remove(screen_video_filename)
