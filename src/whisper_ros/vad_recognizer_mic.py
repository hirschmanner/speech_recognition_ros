#!/usr/bin/env python


import collections
import contextlib
import sys
import wave
import alsaaudio


import webrtcvad
#import speech_recognition as sr

#from std_msgs.msg import Bool as bool_msg
#from std_msgs.msg import Time as time_msg
#from ralli_speech_recognition.msg import RecognizedPhrase

import numpy as np

import whisperx
import time


class Buffer:
    """ The buffer contains the raw audio as bytes in data. It will also keep the startTime and endTime of the recorded
        audio in ROS time.
    """
    def __init__(self, data, startTime, endTime):
        self.data = data
        # self.startTime = time_msg()
        # self.endTime = time_msg()

    def clearBuffer(self):
        self.data = b''
        #self.startTime.data.secs = 0
        #self.startTime.data.nsecs = 0
        #self.endTime.data.secs = 0
        #self.endTime.data.nsecs = 0


class SpeechPublisher:
    """ This class will publish speech recognition results. It uses the data of the WebRTC Voice Activity Detector (VAD)
        to check if there is voice in the recorded audio frames of 10ms. If audio is detected, it will send the buffered
        data to Google Automatic Speech Recognition software and publish the Phrase as well as the start and endtime as
        a ROS message. It will also publish on the topic "ralli_speech_recognition/is_recording" if it is currently
        recording so it can be displayed to the user.
    """
    def __init__(self, framerate=16000, aggressiveness=1, languageString = 'de-DE', maximumDelay = 50, ringBufferSize = 75):
        self.buffer = Buffer(b'', 0, 0)
        self.framerate = framerate
        self.languageString = languageString
        self.audioInput = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL)  # open alsaaudio capture
        self.audioInput.setchannels(1)  # 1 channel
        self.audioInput.setrate(self.framerate)  # set sampling freq
        self.audioInput.setformat(alsaaudio.PCM_FORMAT_S16_LE)  # set 2-byte sample
        self.audioInput.setperiodsize(framerate//100) # Each period has 10ms
        self.vad = webrtcvad.Vad(mode=aggressiveness)
        # self.recognizer = sr.Recognizer()
        # self.pub_phrase = rospy.Publisher('recognized_phrase', RecognizedPhrase, queue_size=10)
        # self.pub_listen = rospy.Publisher("ralli_speech_recognition/is_recording", bool_msg, queue_size=2)
        # self.recognizedPhraseMsg = RecognizedPhrase()
        device = "cuda" 
        compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.batch_size = 4 # used to be 16 - reduce if low on GPU mem
        self.model = whisperx.load_model("small", device, compute_type=compute_type, language='en')
        # self.listenMsg = bool_msg()
        self.maximumDelay = maximumDelay
        self.currentDelay = maximumDelay

        self.ringBufferSize = ringBufferSize
        self.ring_buffer_data = collections.deque(maxlen=ringBufferSize)
        self.ring_buffer_is_speech = collections.deque(maxlen=ringBufferSize)

    def listening_loop(self):
        while True:
            try:
                l, data = self.audioInput.read()  # read data from buffer

                # Check if the data is dected as speech
                is_speech = self.vad.is_speech(data[:], self.framerate)

                # If it is speech add it to the buffer. If it is not, check if there is more than
                # 1 second of recording in the buffer. If it is, send it to Google ASR to recognize phrase
                if is_speech:
                    #if self.buffer.startTime.data.secs == 0:
                    #    self.buffer.startTime.data = rospy.get_rostime()
                    if self.currentDelay != 0:
                        self.currentDelay = 0
                    self.buffer.data += data
                    # self.listenMsg = True
                    # print("SPEECH")
                else:
                    #self.listenMsg = False

                    # We wait for maximumDelay frames (e.g. 50 * 10ms = 500ms) after no voice was detected.
                    # If a new frame with voice is detected, this timer is reset.
                    # A solution with a ring buffer like in vad collector might be more elegant
                    if self.currentDelay < self.maximumDelay:
                        self.buffer.data += data
                        self.currentDelay += 1
                    elif len(self.buffer.data) > 1.0 * self.framerate:
                        # self.buffer.endTime.data = rospy.get_rostime()
                        write_wave('./test.wav', self.buffer.data, self.framerate)
                        # rospy.signal_shutdown('Succesfully written audio')
                        print(len(self.buffer.data))
                        # test=sr.AudioData(buffer, 16000, 2)
                        # write_wave('./test.wav', test.frame_data, 16000)
                        try:
                            # recognizedString = self.recognizer.recognize_google(sr.AudioData(self.buffer.data, 16000, 2),
                            #                                                     None, self.languageString)

                            result = self.model.transcribe(self.buffer.data, batch_size=self.batch_size)
                            recognizedString = result["segments"] # before alignment
                            
                            print(recognizedString)
                            self.recognizedPhraseMsg.recognized_phrase = recognizedString
                            self.recognizedPhraseMsg.start_time = self.buffer.startTime
                            self.recognizedPhraseMsg.end_time = self.buffer.endTime
                            print(self.recognizedPhraseMsg)

                            #self.pub_phrase.publish(self.recognizedPhraseMsg)
                        except:
                            print("Something went wrong.")
                        # except sr.UnknownValueError as e:
                        #     print("Google ASR did not understand anything" + str(e))
                        # except sr.RequestError as e:
                        #     print("Request Error: " + str(e))



                        self.buffer.clearBuffer()
            except KeyboardInterrupt:
                print("Graceful Shutdown")




    def listening_loop_ringbuffer(self):
        """ This function runs in a loop until rospy is shutdown.
            It utilizes two ringbuffers that are filled constantly with the raw audio data and the Booleans if a
            corresponding audio frame is detected as Speech. If more than 50% of the RingBuffer is filled with Speech,
            it will switch into the recording state (triggered = True). The whole RingBuffer of data will be written to
            the buffer and then cleared. The RingBuffer with the Booleans will be reset. In this recording state,
            all new audio is also added to the buffer. The RingBuffer with the Booleans is filled. If less than 50%
            of the Boolean Ringbuffer is Speech, than stop the recording state and send the whole buffer to Google
            Speech Recognition. This way, we have a bit of audio before the detection of speech and also after the
            end of detection of speech. It will also bridge small pauses in speech.
        """
        triggered = False
        file_counter = 0

        try:
            while True:
                l, data = self.audioInput.read()  # read data from buffer

                # Check if the data is dected as speech
                if len(data) >= 320:
                    is_speech = self.vad.is_speech(data[:320], self.framerate) #TODO: Fix weird hack to just go to 320 
                else:
                    continue

                # self.pub_listen.publish(bool_msg(triggered))

                if not triggered:
                    self.ring_buffer_data.append(data[:])
                    self.ring_buffer_is_speech.append(is_speech)

                    # If we're NOTTRIGGERED and more than 70% of the frames in
                    # the ring buffer are voiced frames, then enter the
                    # TRIGGERED state.
                    if np.mean(self.ring_buffer_is_speech) > 0.8:
                        # Set startTime to currentTime - ringBufferSize*0.01s. Each of our samples is 10ms
                        #self.buffer.startTime.data = rospy.get_rostime() - rospy.Time.from_sec(self.ringBufferSize*0.01)
                        triggered = True
                        # We want to yield all the audio we see from now until
                        # we are NOTTRIGGERED, but we have to start with the
                        # audio that's already in the ring buffer.
                        for f in self.ring_buffer_data:
                            self.buffer.data += f

                        self.ring_buffer_data.clear()
                        self.ring_buffer_is_speech.clear()
                else:
                    # We're in the TRIGGERED state, so collect the audio data
                    # and add it to the ring buffer.
                    self.buffer.data += data[:]
                    self.ring_buffer_is_speech.append(is_speech)

                    # If more than 70% of the frames in the ring buffer are
                    # unvoiced, then enter NOTTRIGGERED and yield whatever
                    # audio we've collected.
                    if np.mean(self.ring_buffer_is_speech) < 0.8:
                        #self.buffer.endTime.data = rospy.get_rostime()
                        triggered = False

                        try:
                            # recognizedString = self.recognizer.recognize_google(sr.AudioData(self.buffer.data, 16000, 2),
                            #                                                     None, self.languageString)

                            start = time.time()       
                            result = self.model.transcribe(np.frombuffer(self.buffer.data, dtype=np.int16).flatten().astype(np.float32) / 32768.0, batch_size=self.batch_size)
                            end = time.time()
                            print("Elapsed Time:", end - start)
                            recognizedString = result["segments"] # before alignment
                            print(recognizedString)
                            # write_wave(f'./test{file_counter%10}.wav', self.buffer.data, self.framerate)
                            # file_counter += 1

                            # self.recognizedPhraseMsg.recognized_phrase = recognizedString
                            # self.recognizedPhraseMsg.start_time = self.buffer.startTime
                            # self.recognizedPhraseMsg.end_time = self.buffer.endTime
                            # print(self.recognizedPhraseMsg.recognized_phrase.encode('utf-8'))
                            #print(self.recognizedPhraseMsg.start_time)
                            #print(self.recognizedPhraseMsg.end_time)

                            #self.pub_phrase.publish(self.recognizedPhraseMsg)
                        # except sr.UnknownValueError as e:
                        #     print("Google ASR did not understand anything" + str(e))
                        # except sr.RequestError as e:
                        #     print("Request Error: " + str(e))
                        except:
                            print("Something went wrong.")

                        self.buffer.clearBuffer()
                        self.ring_buffer_is_speech.clear()
        except KeyboardInterrupt:
            print("Graceful Shutdown")



def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def main(args):


    # Optionally, set its aggressiveness mode, which is an integer between 0 and 3. 0 is the least aggressive about
    # filtering out non-speech, 3 is the most aggressive.
    speechRecognition = SpeechPublisher(framerate=16000, aggressiveness=2, languageString='de-DE')
    print("Model Loaded. Speech Recognition Ready!")
    speechRecognition.listening_loop_ringbuffer()



if __name__ == '__main__':
    main(sys.argv[1:])
