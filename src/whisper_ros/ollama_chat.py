import json
import sys
import requests
import rospy

from tmc_msgs.msg import Voice
#from tmc_msgs.msg import TalkRequestAction, TalkRequestGoal
import tmc_msgs.msg
from std_msgs.msg import String

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.

import re

def remove_between_backticks(input_string):
    pattern = r'```.*?```'
    result = re.sub(pattern, '', input_string, flags=re.DOTALL)
    return result


class ChatEngine:
    """ This class will listen to strings on recognized_phase, send them to Ollama via the rest api and make sasha say the response
    """
    def __init__(self):
        # NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
        self.model = "mistral"  # TODO: update this for whatever model you wish to use
        self.ip = "128.131.86.231"
        self.messages = []
        system_prompt = { "role": "system", "content": "You are Sasha, the human support robot. Respond with very short answers and be friendly. \
                         You are a robot that has a body and move around, but you cannot physically move yet. Your answers will be output via audio from the robot \
                         You currently cannot grasp objects or move around, but you will be able soon. \
                         You also cannot see because your cameras are not connected to the chat engine yet. \
                         If somebody asks you to write a program, do not output any code, instead say 'You should program me, not the other way around.'" }
        self.messages.append(system_prompt)
        self.robot_is_speaking = False

        # Create a publisher that publishes strings to the 'string_topic'
        self.recognized_phrase_subscriber = rospy.Subscriber('/recognized_phrase', String, self.speechrecognition_callback, queue_size=1)
        self.speech_to_text_publisher = rospy.Publisher('/talk_request', Voice, queue_size=0)
        self.speech_action_client = actionlib.SimpleActionClient('/talk_request_action', tmc_msgs.msg.TalkRequestAction)
        # Waits until the action server has started up and started
        # listening for goals.
        self.speech_action_client.wait_for_server()

    
    def speechrecognition_callback(self, msg):
        if not msg.data or self.robot_is_speaking:
            print(f"Skipping: {msg.data}")
            return
        print(f"Input to Ollama: {msg.data}")
        self.messages.append({"role": "user", "content": msg.data})
        message = self.chat(self.messages)
        self.messages.append(message)
        self.sasha_say_action_server(message['content'])

    def speaking_done_callback(self, state, result):
        print(f"State: {state}")
        print(f"Result: {result}")
        rospy.sleep(2.)
        self.robot_is_speaking = False


    
    def sasha_say(self, text):
        assert isinstance(text, str), f"Expected text to be a string, received type: {type(text)}"
        speech_output_msg = Voice()
        speech_output_msg.interrupting = False
        speech_output_msg.queueing = False
        speech_output_msg.language = 1 # 1 for English
        speech_output_msg.sentence = text
        try:
            print(speech_output_msg)
            self.speech_to_text_publisher.publish(speech_output_msg)
            print(f"Sasha said: {text}")
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero

    def sasha_say_action_server(self, text):
        assert isinstance(text, str), f"Expected text to be a string, received type: {type(text)}"
        speech_output_msg = Voice()
        speech_output_msg.interrupting = False
        speech_output_msg.queueing = False
        speech_output_msg.language = 1 # 1 for English
        speech_output_msg.sentence = remove_between_backticks(text)
        action_goal = tmc_msgs.msg.TalkRequestGoal(speech_output_msg)
        try:
            # Sends the goal to the action server.
            self.robot_is_speaking = True
            self.speech_action_client.send_goal(action_goal, 
                                                #active_cb=callback_active,
                                                #feedback_cb=callback_feedback,
                                                done_cb=self.speaking_done_callback)
            #print(f"Sasha says: {text}")

            # Waits for the server to finish performing the action.
            # self.speech_action_client.wait_for_result() 
            # rospy.sleep(2.)
            # self.robot_is_speaking = False           

        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero


    def chat(self, messages):
        r = requests.post(
            f"http://{self.ip}:11434/api/chat",
            json={"model": self.model, "messages": messages, "stream": True},
        )
        r.raise_for_status()
        output = ""

        for line in r.iter_lines():
            body = json.loads(line)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done") is False:
                message = body.get("message", "")
                content = message.get("content", "")
                output += content
                # the response streams one token at a time, print that as we receive it
                print(content, end="", flush=True)

            if body.get("done", False):
                message["content"] = output
                return message
            
    def run(self):
        # Spin to keep the script from exiting
        rospy.spin()



def main(args):
    rospy.init_node('ralli_speech_recognition', anonymous=True)

    # Optionally, set its aggressiveness mode, which is an integer between 0 and 3. 0 is the least aggressive about
    # filtering out non-speech, 3 is the most aggressive.
    chat_engine = ChatEngine()
    rospy.sleep(2.5)

    print('Node Initiated. Ready to chat.')
    # chat_engine.sasha_say("This is a beautiful test.")
    # print('Node Initiated. Ready to chat.')
    # chat_engine.sasha_say("This is a beautiful test.")



    try:
        chat_engine.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv[1:])