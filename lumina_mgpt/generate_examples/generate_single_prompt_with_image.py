import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse
from PIL import Image
import torch
from inference_solver import FlexARInferenceSolver
sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])
from xllmx.util.misc import random_seed
import time
from jacobi_utils_static import renew_pipeline_sampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--task", type=str, default='t2i',
                        choices=['t2i', 'depth', 'canny', 'hed', 'openpose', 'i2t'],
                        help="Type of task to perform")
    parser.add_argument("--input_image_path", type=str, default=None,
                        help="Path to the input image (required for depth, canny, hed, openpose)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for image generation")
    parser.add_argument("--speculative_jacobi", default=False, action='store_true')
    parser.add_argument("--quant", default=False, action='store_true')

    args = parser.parse_args()

    print("args:\n", args)
    
    # Default prompt if none provided
    if args.prompt is None:
        args.prompt = "Image of a dog playing water, and a water fall is in the background."
        print(f"Using default prompt: {args.prompt}")

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height
    device = torch.device("cuda")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    image_requiring_tasks = ['depth', 'canny', 'hed', 'openpose', 'i2t']
    if args.task in image_requiring_tasks:
        if args.input_image_path is None:
            raise ValueError(f"Task '{args.task}' requires an input image. Please provide --input_image_path.")
        
        # Verify image exists
        if not os.path.exists(args.input_image_path):
            raise ValueError(f"Input image not found at path: {args.input_image_path}")
        
        print(f"Using input image: {args.input_image_path}")


    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )
    print("checkpoint load finished")

    if args.speculative_jacobi:
        print(inference_solver.__class__)
        print("Use Speculative Jacobi Decoding to accelerate!")
        max_num_new_tokens = 16
        multi_token_init_scheme = 'random' # 'repeat_horizon'
        inference_solver = renew_pipeline_sampler(
            inference_solver,
            jacobi_loop_interval_l = 3,
            jacobi_loop_interval_r = (h // 8)**2 + h // 8 - 10,
            max_num_new_tokens = max_num_new_tokens,
            guidance_scale = cfg,
            seed = None,
            multi_token_init_scheme = multi_token_init_scheme,
            do_cfg=True,
            image_top_k=top_k, 
            text_top_k=10,
            prefix_token_sampler_scheme='speculative_jacobi',
            is_compile=args.quant
        )

    with torch.no_grad():
        for repeat_idx in range(n):
            random_seed(repeat_idx)
            if args.task == 't2i':
                generated = inference_solver.generate(
                        images=[],
                        qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{args.prompt}", None]],  # high-quality synthetic superior
                        max_gen_len=10240,
                        temperature=t,
                        logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                    )
            else:
                task_dict = {"depth": "depth map", "canny": "canny edge map", "hed": "hed edge map", "openpose":"pose estimation map"}
            
                input_image = Image.open(args.input_image_path).convert('RGB')
                generated = inference_solver.generate(
                        images=[input_image],
                        qas=[[f"Generate a dual-panel image of {w}x{h} where the <lower half> displays a <{task_dict[args.task]}>, while the <upper half> retains the original image for direct visual comparison:\n{args.prompt}" , None]], 
                        max_gen_len=10240,
                        temperature=t,
                        logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                        )
            
            output_path = os.path.join(args.save_path, f"output_{repeat_idx}.png")
            generated[1][0].save(output_path)
            print(f"Saved output to {output_path}")
