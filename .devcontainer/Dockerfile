FROM ubuntu:resolute
ARG USERNAME=dev
ARG REPO=/ws

# install rust dependencies, zsh, sudo and git for convenience
ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; apt update; apt install -y \
        ca-certificates libc6-dev pkg-config libssl-dev \
        linux-tools-common linux-tools-generic \
        graphviz clang zsh sudo git wget; \
    apt autoremove -y; apt clean; rm -rf /var/lib/apt/lists/*

# delete default user on new ubuntu images
RUN set -eux; \
    if id ubuntu >/dev/null 2>&1; then \
        grep ubuntu /etc/passwd && \
        touch /var/mail/ubuntu && \
        chown ubuntu /var/mail/ubuntu && \
        userdel -r ubuntu \
    ; fi

# create non-root sudo user
RUN set -eux; \
    useradd --create-home --user-group --no-log-init "$USERNAME"; \
    touch "/etc/sudoers.d/$USERNAME"; \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"; \
    chmod 0440 "/etc/sudoers.d/$USERNAME"

# create workspace dir and fix ownership
RUN set -eux; \
    mkdir -p "$REPO"; \
    chown -R "$USERNAME:$USERNAME" "$REPO" "/home/$USERNAME"
WORKDIR ${REPO}

# change user
USER ${USERNAME}

# install rust
ENV CARGO_HOME="/home/$USERNAME/.cargo"
RUN set -eux; wget -qO - https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# install go
COPY --from=golang:1.26 --chown="$USERNAME:$USERNAME" /usr/local/go/ /usr/local/go/

# add to path
ENV PATH="/home/$USERNAME/.local/bin:$CARGO_HOME/bin:/home/$USERNAME/go/bin:/usr/local/go/bin:$PATH"

# install pprof
RUN set -eux; go install github.com/google/pprof@latest

# install cargo-binstall
RUN set -eux; cargo install --locked cargo-binstall

# install various tools through binstall
RUN set -eux; cargo binstall --locked \
    just just-lsp cargo-nextest cargo-show-asm

# replace entrypoint
CMD sleep infinity