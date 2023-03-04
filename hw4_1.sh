#!/bin/bash
start=$(date +%s)
python3 DirectVoxGO/run.py --config ./DirectVoxGO/configs/nerf/hotdog.py --path_to_json $1 --path_to_imgdir $2 --render_test --eval_ssim --dump_images --render_only
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
# TODO - run your inference Python3 code