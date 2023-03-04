#!/bin/bash
pip install einops ninja
mkdir dvgo_hotdog
wget https://www.dropbox.com/s/nxa5uy6qbjpkdl0/fine_last.tar?dl=1 -O dvgo_hotdog/fine_last.tar
wget https://www.dropbox.com/s/1kt868616ersuhm/coarse_last.tar?dl=1 -O dvgo_hotdog/coarse_last.tar
wget https://www.dropbox.com/s/5u9ilj0i2mvuilg/hw4_2_C_30.ckpt?dl=1 -O hw4_2_C_30.ckpt
# TODO - run your inference Python3 code