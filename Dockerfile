FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set time zone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install openslide-tools
RUN : \
    && apt-get update \
    && apt-get install -y build-essential \
    && apt-get -y install libvips42 \
    && apt-get -y install libgeos-dev \
    && apt-get install -y openslide-tools

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user load_models.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user preprocessing.py /opt/app/

#ENTRYPOINT ["python", "inference.py"]

# Install wholeslidedata
RUN pip install -r /opt/app/requirements.txt
CMD ["/bin/bash"]



