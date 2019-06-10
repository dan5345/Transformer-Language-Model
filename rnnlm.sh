#!/bin/sh


python train.py --data-dir "corpus.bpe.en" \
		--valid-dir "valid" \
		--test-dir "" \
		--mode "train" 
