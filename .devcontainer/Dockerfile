FROM ubuntu:noble
ARG USERNAME=dev
ARG REPO=/ws

# install rust dependencies, zsh, sudo and git for convenience
ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; apt update; apt full-upgrade -y; \
    apt install -y ca-certificates libc6-dev pkg-config libssl-dev \
        linux-tools-common linux-tools-generic linux-tools-"$(uname -r)" \
        graphviz clang zsh sudo git wget; \
    apt autoremove -y; apt clean; rm -rf /var/lib/apt/lists/*

# delete default user on new ubuntu images
RUN set -eux; \
    grep ubuntu /etc/passwd && \
    touch /var/mail/ubuntu && \
    chown ubuntu /var/mail/ubuntu && \
    userdel -r ubuntu

# create non-root sudo user
RUN set -eux; \
    useradd --create-home --user-group --no-log-init "$USERNAME"; \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"; \
    chmod 0440 "/etc/sudoers.d/$USERNAME"

# create workspace dir and fix ownership
RUN set -eux; \
    mkdir -p "$REPO"; \
    chown -R "$USERNAME:$USERNAME" "$REPO" "/home/$USERNAME"
WORKDIR "$REPO"

# change user
USER "$USERNAME"

# install rust
ENV CARGO_HOME="/home/$USERNAME/.cargo"
RUN set -eux; wget -qO - https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly

# install go
COPY --from=golang:1.24-bookworm --chown="$USERNAME:$USERNAME" /usr/local/go/ /usr/local/go/

# add to path
ENV PATH="/home/$USERNAME/.local/bin:$CARGO_HOME/bin:/home/$USERNAME/go/bin:/usr/local/go/bin:$PATH"

# install pprof
RUN set -eux; go install github.com/google/pprof@latest

# replace entrypoint
CMD sleep infinity