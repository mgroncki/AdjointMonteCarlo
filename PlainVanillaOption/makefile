OUT_DIR := bin/

.PHONY: all clean sample

sample:
	mkdir -p $(OUT_DIR)
	g++ -W -Wall -O3 -std=c++11 -o $(OUT_DIR)sample firstSampleCppAd.cpp

clean:
	rm -rf $(OUT_DIR)

all:
	mkdir -p $(OUT_DIR)
	g++ -W -Wall -O3 -std=c++11 -lQuantLib -o $(OUT_DIR)blackMC blackMC.cpp
	./$(OUT_DIR)/blackMC
