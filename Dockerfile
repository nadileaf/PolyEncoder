FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

COPY requirements.txt /opt/requirements.txt
WORKDIR /opt

RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY . /opt

RUN python -m grpc_tools.protoc -Iprotos --python_out=. --grpc_python_out=. protos/*

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION cpp
ENV PYTHONUNBUFFERED 1
CMD python grpc_server.py
