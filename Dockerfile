# * Copyright (c) 2009-2020. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
# RUN --gpus all nvidia/cuda:11.4.0-base-ubuntu18.04 nvidia-smi

FROM cytomine/software-python3-base:v2.2.0
FROM cytomine/software-python3-base

#INSTALL
RUN pip install torch
RUN pip install torchvision
RUN pip install numpy
RUN pip install shapely

RUN mkdir -p /models 
ADD /models/cervicare-2class_dn21adam_best_model_100ep.pth /models/cervicare-2class_dn21adam_best_model_100ep.pth
RUN chmod 444 /models/cervicare-2class_dn21adam_best_model_100ep.pth

#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["python3", "/app/run.py"]
