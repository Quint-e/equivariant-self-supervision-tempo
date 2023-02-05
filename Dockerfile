FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /opt/ml/code/

RUN apt update && \
    apt install -y curl && \
    apt install unzip && \
    apt install ffmpeg -y && \
    apt install libsndfile1-dev -y && \
    apt install -y build-essential

# Install torchaudio 
RUN pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install other requirements  
COPY ./requirements.txt  /opt/ml/code/requirements.txt   
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt 

# Copy and install code
COPY ./ /opt/ml/code/
RUN pip install -e . #Install sst in editable mode.

WORKDIR /opt/ml/code/sst/

ENTRYPOINT ["python"]
