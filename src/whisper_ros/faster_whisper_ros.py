#!/usr/bin/env python


import collections
import contextlib
import sys
import wave
import rospy

import webrtcvad
#import speech_recognition as sr

from std_msgs.msg import Bool as bool_msg
from std_msgs.msg import Time as time_msg
from audio_common_msgs.msg import AudioData
from std_srvs.srv import SetBool, SetBoolResponse

from std_msgs.msg import String

#from ralli_speech_recognition.msg import RecognizedPhrase

import numpy as np

from faster_whisper import WhisperModel, available_models
import torch

import time

language_dict = {
    "en": "en",  # English
    "fr": "fr",  # French
    "de": "de",  # German
    "es": "es",  # Spanish
    "it": "it",  # Italian
    "ja": "ja",  # Japanese
    "zh": "zh",  # Chinese
    "nl": "nl",  # Dutch
    "uk": "uk",  # Ukrainian
    "pt": "pt",  # Portuguese
    "german": "de",    # German
    "english": "en",   # English
    "french": "fr",    # French
    "spanish": "es",   # Spanish
    "italian": "it",   # Italian
    "japanese": "ja",  # Japanese
    "chinese": "zh",   # Chinese
    "dutch": "nl",     # Dutch
    "ukrainian": "uk", # Ukrainian
    "portuguese": "pt" # Portuguese
}


class Buffer:
    """ The buffer contains the raw audio as bytes in data. It will also keep the startTime and endTime of the recorded
        audio in ROS time.
    """
    def __init__(self, data, startTime, endTime):
        self.data = data
        self.startTime = time_msg()
        self.endTime = time_msg()

    def clearBuffer(self):
        self.data = b''
        self.startTime.data.secs = 0
        self.startTime.data.nsecs = 0
        self.endTime.data.secs = 0
        self.endTime.data.nsecs = 0


class SpeechPublisher:
    """ This class will publish speech recognition results. It uses Voice Activity Detector (VAD)
        to check if there is voice in the recorded audio. If audio is detected, it will send the buffered
        data to Whisper and publish the Phrase as a ROS message. 
    """
    def __init__(self,
                 framerate=16000, 
                 whisper_model = 'large-v3-turbo',
                 languageString = 'en',
                 ring_buffer_size = 1000, 
                 vad = "silero", 
                 speech_threshold=0.7): 
        
        self.write_wav_files = False # We can write wav files of the audio chunks. Mostly used for debugging
        if self.write_wav_files:
            self.file_counter = 0
        
        # The main buffer that will be fed into the speech recognition model
        self.buffer = Buffer(b'', 0, 0)
        self.framerate = framerate
        self.languageString = languageString
        self.vad_type = vad
        self.speech_threshold = speech_threshold
        
        if self.vad_type == 'webrtc':
            self.vad_webrtc = webrtcvad.Vad(mode=2) # The aggressiveness handles how aggressive the VAD is to remove non-speech. Between 0 and 3
            if framerate in [8000, 16000, 32000, 48000]:
                self.vad_chunk_size = framerate * 0.01 * 2 # 10 ms for 16 bit uint
            else:
                rospy.logerr(f"Invalid framerate '{framerate}'! Must be one of [8000, 16000, 32000, 48000].")
                raise ValueError(f"Invalid framerate '{framerate}'. Must be one of [8000, 16000, 32000, 48000].")
            self.vad_chunk_size = 320
        elif self.vad_type == 'silero':
            if framerate == 16000:
                self.vad_chunk_size = 1024
            elif framerate == 8000:
                self.vad_chunk_size = 512
            else:
                rospy.logerr(f"Invalid framerate '{framerate}'! For Silero VAD, must be one of [8000, 16000].")
                raise ValueError(f"Invalid framerate '{framerate}'. For Silero VAD, must be one of [8000, 16000].")

            self.vad_silero, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False)
            

        device = "cuda" 
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        
        model_size = whisper_model
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)


        #self.listenMsg = bool_msg()
        #self.vad_chunk_size = 320
        msg = rospy.wait_for_message("audio/audio", AudioData)
        self.audio_message_length_bits = len(msg.data)
        self.audio_message_length_time = len(msg.data)/(2*self.framerate)

        self.data_buffer_size = int(np.ceil(ring_buffer_size / (self.audio_message_length_time*1000)))

        if self.vad_chunk_size <= self.audio_message_length_bits:
            self.is_speech_buffer_size = int(self.data_buffer_size * (self.audio_message_length_bits // self.vad_chunk_size))
        else:
            self.is_speech_buffer_size = int(self.data_buffer_size // np.ceil(self.vad_chunk_size / self.audio_message_length_bits))
            

        self.ring_buffer_data = collections.deque(maxlen=self.data_buffer_size)
        self.ring_buffer_is_speech = collections.deque(maxlen=self.is_speech_buffer_size)
        self.vad_buffer = b''
        self.triggered = False


        self.pause_speech_recognition = rospy.Service('pause_speech_recognition', SetBool, self.pause_speech_recognition)
        self.is_paused = False

        self.start_counter = False
        self.counter = 0
        # Create a publisher that publishes strings to the 'string_topic'
        self.recognized_phrase_publisher = rospy.Publisher('/recognized_phrase', String, queue_size=10)
        self.subscriber = rospy.Subscriber('/audio/audio', AudioData, self.audio_callback, queue_size=10)

    
    def audio_callback(self, msg):
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
        #start_time = time.time()
        if self.is_paused:
            if len(self.vad_buffer)>0:
                self.vad_buffer = b''
            if len(self.ring_buffer_data)>0:
                self.ring_buffer_data.clear()
            if len(self.ring_buffer_is_speech)>0:
                self.ring_buffer_is_speech.clear()
            if len(self.buffer.data):
                self.buffer.clearBuffer()
            return
        data = msg.data

        # We collect enough chunks in the vad_buffer until they are ready to be analyzed by VAD.
        self.vad_buffer+=data
        if len(self.vad_buffer)>=self.vad_chunk_size:

            # There might be multiple chunks in one audio message, so we iterate over them.
            for i in range(0, len(self.vad_buffer), self.vad_chunk_size):

                # If the data is too short, we skip it.
                if len(self.vad_buffer[i:i + self.vad_chunk_size]) < self.vad_chunk_size:
                    break

                # Depending on if we are using Silero or WebRTC we check if there is speech in the audio chunk.
                # If we found speech, the respective bool in self.ring_buffer_is_speech is set to True.
                if self.vad_type == 'silero':
                    audio_float32 = np.frombuffer(self.vad_buffer[i:i + self.vad_chunk_size], dtype=np.int16).flatten().astype(np.float32) / 32768.0
                    new_confidence = self.vad_silero(torch.from_numpy(audio_float32), 16000).item()
                    self.ring_buffer_is_speech.append(new_confidence>0.5)
                elif self.vad_type == 'webrtc':
                    is_speech = self.vad_webrtc.is_speech(self.vad_buffer[i:i + self.vad_chunk_size], self.framerate)
                    self.ring_buffer_is_speech.append(is_speech)
            self.vad_buffer = b''
            
        # While the speech recognition is not triggered, we collect the audio chunks in the ring buffer.
        if not self.triggered:
            self.ring_buffer_data.append(data[:])

            # If we're NOTTRIGGERED and more than 'speech_threshold' of the frames in
            # the ring buffer are voiced frames, then enter the TRIGGERED state.
            if len(self.ring_buffer_is_speech) == self.is_speech_buffer_size:
                if np.mean(self.ring_buffer_is_speech) > self.speech_threshold:
                    # Set startTime to currentTime - ringBufferSize*0.01s. Each of our samples is 10ms
                    self.buffer.startTime.data = rospy.get_rostime() - rospy.Time.from_sec(self.data_buffer_size*self.audio_message_length_time/self.framerate)
                    self.triggered = True
                    print("TRIGGERED")
                    # We want to yield all the audio we see from now until we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer and move it to the main buffer.
                    for f in self.ring_buffer_data:
                        self.buffer.data += f

                    self.ring_buffer_data.clear()
                    self.ring_buffer_is_speech.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the main buffer.
            self.buffer.data += data[:]

            # If more than 70% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if len(self.ring_buffer_is_speech) == self.is_speech_buffer_size:
                if np.mean(self.ring_buffer_is_speech) < 0.7:
                    self.buffer.endTime.data = rospy.get_rostime()
                    self.triggered = False
                    print("Untriggered")

                    try:
                        segments, info = self.model.transcribe(np.frombuffer(self.buffer.data, dtype=np.int16).flatten().astype(np.float16) / 32768.0, beam_size=5, language=self.languageString)
                        result = ""
                        for segment in segments:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                            result += segment.text
                        recognizedString = result

                        if recognizedString:
                            # Create a string message
                            message = String()                   
                            message.data = recognizedString

                            # Publish the string message
                            self.recognized_phrase_publisher.publish(message)
                        
                        if self.write_wav_files:
                            write_wave(f'./test_ros{self.file_counter%10}.wav', self.buffer.data, self.framerate)
                            self.file_counter += 1

                    except Exception as error:
                        # handle the exception
                        print("An exception occurred:", error) # An exception occurred: division by zero

                    self.buffer.clearBuffer()
                    self.ring_buffer_is_speech.clear()

    def pause_speech_recognition(self, req):
        self.is_paused = req.data
        if req.data:
            print(f"Paused speech recognition.")
            return SetBoolResponse(success = True, message = f"Paused speech recognition.")
        else:
            print(f"Unpaused speech recognition.")
            return SetBoolResponse(success = True, message = f"Unpaused speech recognition.")

    def run(self):
        # Spin to keep the script from exiting
        rospy.spin()



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
        wf.setnchannels(1) # Mono
        wf.setsampwidth(2) # 16-bit samples
        wf.setframerate(sample_rate) # 16kHz sample rate
        wf.writeframes(audio)


def main(args):
    rospy.init_node('speech_recognition', anonymous=True)

    language = rospy.get_param('~language', 'english')
    if not language in language_dict:
        rospy.logerr(f"Invalid language '{language}'! Must be one of {list(language_dict.keys())}")
        raise ValueError(f"Invalid language '{language}'. Allowed values are: {list(language_dict.keys())}")
    
    whisper_model = rospy.get_param('~whisper_model', 'large-v3-turbo')
    if not whisper_model in available_models():
        rospy.logerr(f"Invalid whisper_model '{whisper_model}'! Must be one of {available_models()}")
        raise ValueError(f"Invalid whisper_model '{whisper_model}'. Allowed values are: {available_models()}")
    
    vad = rospy.get_param('~vad', 'silero')
    if not vad in ["silero", "webrtc"]:
        rospy.logerr(f"Invalid VAD '{vad}'! Must be one of {['silero', 'webrtc']}")
        raise ValueError(f"Invalid VAD '{vad}'. Allowed values are: {['silero', 'webrtc']}")
    
    buffer_size = int(rospy.get_param('~buffer_size', '1000'))
    if not buffer_size > 0:
        rospy.logerr(f"Invalid buffer_size '{buffer_size}'! Must be positive integer.")
        raise ValueError(f"Invalid buffer_size '{buffer_size}'. Must be positive integer.")
    
    speech_threshold = float(rospy.get_param('~speech_threshold', '0.5'))
    if not speech_threshold >= 0. or not speech_threshold <= 1.0:
        rospy.logerr(f"Invalid speech_threshold '{speech_threshold}'! Must be in interval [0., 1.].")
        raise ValueError(f"Invalid speech_threshold '{speech_threshold}'. Must be in interval [0., 1.].")
    
    framerate = int(rospy.get_param('~framerate', '16000'))
    if not framerate in [8000, 16000, 32000, 48000]:
        rospy.logerr(f"Invalid framerate '{framerate}'! Must be one of [8000, 16000, 32000, 48000].")
        raise ValueError(f"Invalid framerate '{framerate}'. Must be one of [8000, 16000, 32000, 48000].")

    rospy.loginfo(f"Chosen Language: {language}")
  
    speechRecognition = SpeechPublisher(framerate=framerate, 
                                        whisper_model=whisper_model,
                                        languageString=language_dict[language],
                                        ring_buffer_size=buffer_size,
                                        vad=vad,
                                        speech_threshold=speech_threshold)
    rospy.sleep(2.)
    print('Node Initiated. Ready to Recognize Speech.')

    try:
        speechRecognition.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
