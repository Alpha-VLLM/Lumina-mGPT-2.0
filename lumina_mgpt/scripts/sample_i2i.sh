# for image to image generation, you need to set the --i2i_task --> depth, canny, subject and give the --image_path and --image_prompt
task="i2i"
i2i_task="depth"
image_path="assets/depth.png"
image_prompt="A rubber outdoor basketball. On a sunlit outdoor court, it bounces near a vibrant mural, casting a long shadow on the asphalt as children eagerly chase it."
cuda_number=0
results_dir=samples/
mkdir -p ${results_dir}

CUDA_VISIBLE_DEVICES=${cuda_number} python generate_examples/generate.py \
    --model_path Alpha-VLLM/Lumina-mGPT-2.0 \
    --save_path ${results_dir} --cfg 4.0 --top_k 4096 --temperature 1.0 --width 512 --height 1024 \
    --task ${task} --i2i_task ${i2i_task} \
    --image_path ${image_path} \
    --image_prompt ${image_prompt} \