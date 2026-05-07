# Makefile for woys-pw-out — native PipeWire output helper.
# v0.9.0 — replaces pw-cat / pacat subprocess on the playback path.
# See bin/woys-pw-out.c for the architectural rationale.
#
# Targets:
#   make            — build the helper as ./woys-pw-out
#   make clean      — remove the built binary
#   make install    — install to $(PREFIX)/bin (default: $$HOME/.local/bin)
#   make smoke      — build + run --help to confirm the binary loads

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11
LDFLAGS ?=
PKGCFG_FLAGS := $(shell pkg-config --cflags libpipewire-0.3)
PKGCFG_LIBS  := $(shell pkg-config --libs   libpipewire-0.3)

BIN := woys-pw-out
SRC := woys-pw-out.c

PREFIX ?= $(HOME)/.local
BINDIR := $(PREFIX)/bin

.PHONY: all clean install smoke

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(PKGCFG_FLAGS) -o $@ $< $(LDFLAGS) $(PKGCFG_LIBS) -lpthread

clean:
	rm -f $(BIN)

install: $(BIN)
	install -Dm755 $(BIN) $(BINDIR)/$(BIN)

smoke: $(BIN)
	./$(BIN) --help 2>&1 | head -10
