FROM python:3.7

WORKDIR /opt

ADD . /opt

RUN pip install -r requirements_cpu.txt

RUN pip install -e .