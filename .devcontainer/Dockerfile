# syntax=docker/dockerfile:1

ARG IMAGE_VARIANT=trixie
ARG RUST_VERSION=1.100
ARG GO_VERSION=1.27


FROM golang:${GO_VERSION} AS pprof
RUN mkdir -p /opt/go && \
    GOPATH=/opt/go go install github.com/google/pprof@latest


FROM rust:${RUST_VERSION}-${IMAGE_VARIANT} AS rust_img


FROM rust_img AS binstall
ARG BINSTALL_URL=https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh
RUN curl -L --proto '=https' --tlsv1.2 -sSf ${BINSTALL_URL} | bash && \
    mkdir -p /opt/binstall && cargo binstall --locked -y --no-track --root /opt/binstall \
    just just-lsp cargo-nextest cargo-show-asm


FROM rust_img AS final
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    graphviz && \
    rm -rf /var/lib/apt/lists/*
ARG USERNAME=dev
RUN useradd --create-home --user-group --no-log-init "$USERNAME"
WORKDIR /home/${USERNAME}
USER ${USERNAME}
RUN rustup component add clippy rustfmt rust-src rust-docs
COPY --from=pprof --chown="$USERNAME:$USERNAME" /opt/go/bin/ /usr/local/bin/
COPY --from=binstall --chown="$USERNAME:$USERNAME" /opt/binstall/bin/ /usr/local/bin/
CMD ["sleep", "infinity"]
