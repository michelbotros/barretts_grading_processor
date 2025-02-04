FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set time zone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common git \
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

# Copy application files before changing user
WORKDIR /opt/app
COPY requirements.txt /opt/app/

# Install Python dependencies as root (before switching to user)
RUN pip install --no-cache-dir -r /opt/app/requirements.txt
RUN pip install --no-cache-dir git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main

# Create a non-root user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

# Copy the rest of the application files
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py load_models.py preprocessing.py /opt/app/

# Set entry point
ENTRYPOINT ["python", "inference.py"]
