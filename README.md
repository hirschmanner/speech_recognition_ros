## ROS Speech Recognition

This package is a simple speech recognition node for ROS Noetic. It uses the OpenAI's Whisper models through the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementation. The default voice activity detection is done using [Silero VAD](https://github.com/snakers4/silero-vad). 

This package is still a work in process and should be used with care.

## Getting Started

The easiest way to use this project is with Docker.

### Docker

Simply execute the commands below to build and run the docker container.
```
cd docker
docker compose build
docker compose run --name speechrecognition-ros whisper-ros
```
In the docker container you can start the speech recognition with:
```
roslaunch whisper-ros speechrecognition_ros.launch
```

### Run Natively

1. Install ROS Noetic as described [here](http://wiki.ros.org/noetic/Installation) and create a catkin workspace.
2. Clone the repo into your catkin_ws/src.
3. Install the pip requirements with 
```
pip install -r requirements.txt
```
4. Build your catkin workspace and launch the speech recognition with 
```
roslaunch whisper-ros speechrecognition_ros.launch
```

## Contact

Matthias Hirschmanner - hirschmanner@acin.tuwien.ac.at


<!--p align="right">(<a href="#readme-top">back to top</a>)</p-->



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project is heavily based on other open-source projects. We want to thank all people involved for their amazing work.

* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* [Silero VAD](https://github.com/snakers4/silero-vad)
* [py-webrtcvad](https://github.com/wiseman/py-webrtcvad)