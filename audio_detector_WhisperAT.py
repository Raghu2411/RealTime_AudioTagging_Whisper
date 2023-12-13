import pyaudio
import whisper_at as whisper
import numpy as np
import torch
import math
import struct
import wave
import time
import datetime
import os
from tempfile import NamedTemporaryFile

# TRIGGER_RMS = 10  # start recording above 10
# RATE = 16000  # sample rate
# TIMEOUT_SECS = 1  # silence time after which recording stops
# FRAME_SECS = 0.25  # length of frame(chunks) to be processed at once in secs
# CUSHION_SECS = 1  # amount of recording before and after sound
#
# SHORT_NORMALIZE = (1.0 / 32768.0)
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# SHORT_WIDTH = 2
# CHUNK = int(RATE * FRAME_SECS)
# CUSHION_FRAMES = int(CUSHION_SECS / FRAME_SECS)
# TIMEOUT_FRAMES = int(TIMEOUT_SECS / FRAME_SECS)

#TRIGGER_RMS = 5
TRIGGER_RMS = 10
# RATE = 44100 # = 300MB/hour
RATE = 22050  # = 150MB/hour
# RATE = 88200  # = 450/hour
# RATE = 16000
TIMEOUT_SECS = 5
FRAME_SECS = 0.25  # length of frame in secs
CUSHION_SECS = 1  # amount of recording before and after sound

SHORT_NORMALIZE = (1.0 / 32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SHORT_WIDTH = 2
CHUNK = int(RATE * FRAME_SECS)
CUSHION_FRAMES = int(CUSHION_SECS / FRAME_SECS)
TIMEOUT_FRAMES = int(TIMEOUT_SECS / FRAME_SECS)

all_seg = []
f_name_directory = '/Users/rac/PycharmProjects/AudioForBest/Recordings/'
audio_model = whisper.load_model("medium")


class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / SHORT_WIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)
        self.time = time.time()
        self.quiet = []
        self.quiet_idx = -1
        self.timeout = 0

    def record(self):
        print('')
        sound = []
        start = time.time()
        begin_time = None
        while True:
            data = self.stream.read(CHUNK)
            rms_val = self.rms(data)
            if self.inSound(data):
                sound.append(data)
                if begin_time == None:
                    begin_time = datetime.datetime.now()
            else:
                if len(sound) > 0:
                    self.write(sound, begin_time)
                    sound.clear()
                    begin_time = None
                else:
                    self.queueQuiet(data)

            curr = time.time()
            secs = int(curr - start)
            tout = 0 if self.timeout == 0 else int(self.timeout - curr)
            label = 'Listening' if self.timeout == 0 else 'Recording'
            print('[+] %s: Level=[%4.2f] Secs=[%d] Timeout=[%d]' % (label, rms_val, secs, tout), end='\r')

    # quiet is a circular buffer of size cushion
    def queueQuiet(self, data):
        self.quiet_idx += 1
        # start over again on overflow
        if self.quiet_idx == CUSHION_FRAMES:
            self.quiet_idx = 0

        # fill up the queue
        if len(self.quiet) < CUSHION_FRAMES:
            self.quiet.append(data)
        # replace the element on the index in a cicular loop like this 0 -> 1 -> 2 -> 3 -> 0 and so on...
        else:
            self.quiet[self.quiet_idx] = data

    def dequeueQuiet(self, sound):
        if len(self.quiet) == 0:
            return sound

        ret = []

        if len(self.quiet) < CUSHION_FRAMES:
            ret.append(self.quiet)
            ret.extend(sound)
        else:
            ret.extend(self.quiet[self.quiet_idx + 1:])
            ret.extend(self.quiet[:self.quiet_idx + 1])
            ret.extend(sound)

        return ret

    def inSound(self, data):
        rms = self.rms(data)
        curr = time.time()

        if rms > TRIGGER_RMS:
            self.timeout = curr + TIMEOUT_SECS
            return True

        if curr < self.timeout:
            return True

        self.timeout = 0
        return False

    def write(self, sound, begin_time):
        # insert the pre-sound quiet frames into sound
        sound = self.dequeueQuiet(sound)

        # sound ends with TIMEOUT_FRAMES of quiet
        # remove all but CUSHION_FRAMES
        keep_frames = len(sound) - TIMEOUT_FRAMES + CUSHION_FRAMES
        recording = b''.join(sound[0:keep_frames])

        # recording = bytes(''.join(str(sound[0:keep_frames])))
        # recording = bytes(''.join(str(sound[0:keep_frames])).encode('utf-8'))
        # recording = bytes(''.join(str(sound[0:keep_frames])).encode('ISO-8859-1'))
        # Change into numpy array
        recording_np = np.frombuffer(recording, dtype=np.int16)  # dtype=np.uint8
        # recording_np = np.frombuffer(recording, np.int16).flatten().astype(np.float32) / 32768.0

        filename = begin_time.strftime('%Y-%m-%d_%H.%M.%S')
        pathname = os.path.join(f_name_directory, '{}.wav'.format(filename))

        # Getting the Segments
        print("Model Loaded and processing.")
        result = audio_model.transcribe(recording_np, fp16=torch.cuda.is_available())
        at_res = whisper.parse_at_label(result, language='en', p_threshold=-1)
        for segment in at_res:
            cur_tags = segment['audio tags']
            all_seg.append(cur_tags)

        # Getting all the violations
        ans = ','.join(str(v) for v in all_seg)
        all_seg.clear()
        print("Got the detected segments and added to the string")

        # print(ans)

        # Save the Audio_File_Name with the Tag in a file
        print("Now writing in the file")
        violation_list_file = open("speechViolationFile.txt", "a")
        violation_list_file.write(filename + " : " + ans+ "\n")
        # violation_list_file.write(filename+"\n")
        violation_list_file.close()
        wf = wave.open(pathname, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('[+] Saved: {}'.format(pathname))


a = Recorder()
a.record()
