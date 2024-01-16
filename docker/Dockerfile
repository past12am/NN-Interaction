FROM gcc:13.2

# Install cmake and ninja
#RUN set -ex;                                                                      \
#    apt-get update;                                                               \
#    apt-get install -y cmake libzmq3-dev ninja-build

# Set workdir and copy src
ADD "./../../" "/usr/src/nn-interaction/"

WORKDIR /usr/src/
RUN ls /usr/src/nn-interaction

# Compile application
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=ninja -G Ninja -S ./nn-interaction -B ./nn-interaction-build

RUN gcc -o myapp main.c
CMD ["./myapp"]