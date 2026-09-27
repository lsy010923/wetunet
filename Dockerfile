FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace/wetbirdnet

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
