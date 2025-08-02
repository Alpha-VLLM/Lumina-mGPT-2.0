# for image pair generation, you need to set the task --> depth, canny, hed, openpose
task="t2i"
cuda_number=0
results_dir=samples/
mkdir -p ${results_dir}

CUDA_VISIBLE_DEVICES=${cuda_number} python generate_examples/generate.py \
    --model_path Alpha-VLLM/Lumina-mGPT-2.0 \
    --save_path ${results_dir} --cfg 4.0 --top_k 4096 --temperature 1.0 --width 768 --height 768 --speculative_jacobi --task ${task}
