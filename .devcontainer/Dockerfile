FROM ubuntu:noble
ARG USERNAME=dev
ARG REPO=/ws

# install mold and clang for faster linking, zsh, sudo and git for convenience
ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; apt update; apt full-upgrade -y; \
    apt install -y ca-certificates libc6-dev pkg-config libssl-dev \
        mold clang zsh sudo git wget; \
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

# add cargo config for multithreaded frontend and use mold linker
COPY fast_cargo_config.toml "/home/$USERNAME/.cargo/config.toml"

# create workspace dir and fix ownership
RUN set -eux; \
    mkdir -p "$REPO"; \
    chown -R "$USERNAME:$USERNAME" "$REPO" "/home/$USERNAME"
WORKDIR "$REPO"

# change user
USER "$USERNAME"

# install rust
RUN set -eux; wget -qO - https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly

# add local bin to path
ENV PATH="/home/$USERNAME/.local/bin:/home/$USERNAME/.cargo/bin:$PATH"

# replace entrypoint
CMD sleep infinity