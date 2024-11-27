FROM huggingface/transformers-pytorch-gpu:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG ROOT_USER=root
ARG ROOT_PW=${ROOT_USER}

#
# Locale Setting
#
ENV LC_ALL C.UTF-8
RUN echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc

#
# Timezone Setting
#
RUN apt-get update && apt-get install -y tzdata
RUN ls /usr/share/zoneinfo && \
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
echo "Asia/Seoul" >  /etc/timezone \

#
# OPENSSH
#
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /run/sshd

#set password
RUN echo "${ROOT_USER}:${ROOT_PW}" | chpasswd

#replace sshd_config
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

EXPOSE 22

#
# Custom Ubuntu APT Packages
#
#RUN apt-get update && apt-get install -y screen htop
RUN apt-get update && apt-get install -y libgtk2.0-common

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#
# Python Packages
#
COPY requirements.txt /tmp
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt


ARG ENVIRONMENT

#
# Entrypoint
#
COPY --chmod=755 docker-entrypoint.${ENVIRONMENT}.sh /docker-entrypoint.sh
ENTRYPOINT "/docker-entrypoint.sh"

###############################################
#
# For Development
#

#
# Add User
#
ARG USER_ID
ARG GROUP_ID
ARG USER
ARG USER_PW

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} ${USER} && \
    useradd -l -m -u ${USER_ID} -g ${USER} ${USER} \
;fi
RUN chmod 777 /root

RUN if [ -n "${USER}" ] && [ -n "${USER_PW}" ]; then \
    echo "${USER}:${USER_PW}" | chpasswd \
;fi


#RUN apt-get update && apt-get install -y screen htop