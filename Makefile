CC = gcc
PYTHON = python
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS = -lm
TARGET = fluid.exe
FRAMES = 5000
SCENE = cylinder

.PHONY: all run deps clean help

all: $(TARGET)

$(TARGET): fluid.c Makefile
	$(CC) $(CFLAGS) fluid.c -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	$(PYTHON) view.py --exe ./$(TARGET) --frames $(FRAMES) --scene $(SCENE)

deps:
	$(PYTHON) -m pip install numpy matplotlib

clean:
	$(PYTHON) -c "from pathlib import Path; Path('$(TARGET)').unlink(missing_ok=True)"

help:
	@echo "make                 Build fluid.exe"
	@echo "make run             Build (if needed) and start the viewer"
	@echo "make run FRAMES=800 SCENE=free"
	@echo "make deps            Install Python dependencies"
	@echo "make clean           Remove fluid.exe"
