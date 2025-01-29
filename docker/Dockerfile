FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04 AS whisper_ros_cudnn

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y software-properties-common
	#python3-opencv 
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get install -y python3.8
RUN apt-get install -y ca-certificates python3-dev python3-tk git wget sudo ninja-build vim
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py


# install lsb-release and curl and stuff for audio
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    lsb-release \
    curl \
    alsa-utils \ 
    portaudio19-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# add the keys
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# install ros
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #ros-noetic-desktop-full \
    ros-noetic-ros-base \
    ros-noetic-catkin \
    #ros-noetic-vision-msgs \
    ros-noetic-audio-common \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
 && apt-get clean 


# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools 

RUN sudo rosdep init
RUN rosdep update
RUN mkdir -p /catkin_ws/src
RUN cd /catkin_ws/
RUN source ~/.bashrc
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'
#RUN catkin_make
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

RUN source ~/.bashrc

# install requirements
COPY docker/requirements.txt .
RUN pip install importlib_metadata --upgrade
RUN pip install -r requirements.txt --ignore-installed PyYAML



WORKDIR /root
