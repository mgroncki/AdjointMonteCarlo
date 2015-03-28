OUT_DIR := bin/

.PHONY: all clean

clean:
	rm -rf $(OUT_DIR)

all:
	mkdir -p $(OUT_DIR)
	g++ -W -Wall -O3 -std=c++11 -o $(OUT_DIR)blackMC blackMC.cpp
