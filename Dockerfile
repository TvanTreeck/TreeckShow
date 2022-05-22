FROM pytorch/pytorch

RUN apt update

RUN apt install -y \
    git \
    vim

COPY requirements.txt .
RUN pip install -r requirements.txt

