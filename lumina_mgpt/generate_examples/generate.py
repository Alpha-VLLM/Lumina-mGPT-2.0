import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse
from PIL import Image
import torch
from inference_solver import FlexARInferenceSolver
sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])
from xllmx.util.misc import random_seed
from xllmx.data.data_reader import read_general
from lumina_mgpt.data.item_processor import center_crop
import time
from jacobi_utils_static import renew_pipeline_sampler

def check_args(args):
    if args.task == 'i2i':
        # 1. Check if image_path is provided
        if args.image_path is None or args.image_prompt is None:
            raise ValueError("Error: --image_path and --image_prompt is required when task is 'i2i'.")

        # 2. Check if the image_path exists in the system
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Error: The specified image path does not exist: '{args.image_path}'")

        # 3. Check if the path points to a file (not a directory)
        if not os.path.isfile(args.image_path):
             raise ValueError(f"Error: The specified image path is not a file: '{args.image_path}'")

        # 4. Check if the file extension conforms to common image naming conventions
        valid_image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff']
        _, file_extension = os.path.splitext(args.image_path)
        if file_extension.lower() not in valid_image_extensions:
            raise ValueError(f"Error: File '{args.image_path}' has an unrecognized image extension '{file_extension}'. "
                             f"Supported extensions: {', '.join(valid_image_extensions)}")
        image = Image.open(read_general(args.image_path))
        w, h = image.size
        if w == args.width and h == args.height // 2:
            print("Size of input image is already half of the full picture.")
        else:
            print(f"Do center crop to reshape the image into {args.width} x {args.height // 2}.")
            new_image = center_crop(image, crop_size=[args.width, args.height // 2])
            new_image_path = args.save_path + f"{args.i2i_task}_ref_center_crop.png"
            new_image.save(new_image_path)
            print(f"Center cropped image saved to {new_image_path}.")
            args.image_path = new_image_path
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for Lumina-mGPT-2.0.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--temperature", type=float, help="Temperature parameter for generation")
    parser.add_argument("--top_k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--cfg", type=float, help="Classifier-Free Guidance (CFG) scale factor")
    parser.add_argument("-n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--width", type=int, default=256, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=256, help="Height of the generated image")
    parser.add_argument("--task", type=str, default='t2i', choices=['t2i', 'i2i', 'depth', 'canny', 'hed', 'openpose'], help="Type of task to perform")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image (for i2i or other image-related tasks)")
    parser.add_argument("--image_prompt", type=str, default=None, help="Prompt for the image in i2i task")
    parser.add_argument("--i2i_task", type=str, default='object_control', choices=['depth', 'canny', 'subject'], help="Specific control type for i2i task")
    parser.add_argument("--speculative_jacobi", default=False, action='store_true', help="Enable Speculative Jacobi decoding, not recommanded for i2i task currently")
    parser.add_argument("--quant", default=False, action='store_true', help="Quantize the model")

    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    try:
        check_args(args)
        print(f"Arguments parsed and checked. Ready to execute task '{args.task}'...")
        if args.task == 'i2i':
            print(f"Execute i2i task {args.i2i_task} use prompt: {args.image_prompt}\nimage: {args.image_path}.")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        parser.print_help()
        exit(1) # Exit with an error code

    print("args:\n", args)
    l_prompts = [
        "Image of a dog playing water, and a water fall is in the background.",
        "A high-resolution photograph of a middle-aged woman with curly hair, wearing traditional Japanese kimono, smiling gently under a cherry blossom tree in full bloom.",  # noqa
        "A pink and chocolate cake ln front of a white background",
        "A highly detailed, 3D-rendered, stylized representation of 2 travellers, a 40 year old man and a step behind, a 40 year old woman walking on a path. Full body visible. They have large, expressive hazel eyes and dark, curly hair that is slightly messy. Their faces are full of innocent wonder, with rosy cheeks and a scattering of light freckles across his nose and cheeks. They are wearing thicker clothes, trousers and hiking shoes, making it look cosy. The background is softly blurred with warm tones, putting full focus on their facial features and expressions. The image has a soft, cinematic lighting style with subtle shadows that highlight the contours of their faces, giving it a realistic yet animated look. The overall art style is similar to modern animated films with high levels of detail and a slight painterly touch, 8k.",
        "Indoor portrait of a young woman with light blonde hair sitting in front of a large window. She is positioned slightly to the right, wearing an oversized white shirt with rolled-up sleeves and brown pants. A black shoulder bag is slung over her left shoulder. Her lips are pursed in a playful expression. The window behind her features closed horizontal blinds, reflecting faint interior lighting. The surrounding wall is made of textured, light-colored brick. The lighting is soft, highlighting her features and the textures of her clothing and the wall. Casual, candid, slightly cool color temperature, natural pose, balanced composition, urban interior environment."
    ]

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height
    device = torch.device("cuda")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )
    print("checkpiont load finished")

    if args.speculative_jacobi:
        print(inference_solver.__class__)
        print("Use Speculative Jacobi Decoding to accelerate!")
        max_num_new_tokens = 16
        multi_token_init_scheme = 'random' # 'repeat_horizon'
        inference_solver = renew_pipeline_sampler(
            inference_solver,
            jacobi_loop_interval_l = 3 if args.task != 'i2i' else 16,
            jacobi_loop_interval_r = (((h // 8) * (w // 8) + h // 8 - 10) if args.task != 'i2i' else ((h // 2 // 8) * (w // 8) + h // 2 // 8 - 10)),
            max_num_new_tokens = max_num_new_tokens,
            guidance_scale = cfg,
            seed = None,
            multi_token_init_scheme = multi_token_init_scheme,
            do_cfg=True,
            image_top_k=top_k, 
            text_top_k=10,
            prefix_token_sampler_scheme='speculative_jacobi',
        )

    with torch.no_grad():
        if args.task == 'i2i':
            task_dict = {"depth": "depth map", "canny": "canny edge map", "subject": "Bubbly and effervescent with a sparkling allure."}
            prompt = f"Generate a dual-panel image of {w}x{h} where the <upper half> displays a <{task_dict[args.i2i_task]}>, while the <lower half> retains the original image for direct visual comparison:\n{args.image_prompt}"
            generated = inference_solver.generate(
                    images=[args.image_path],
                    qas=[[prompt, "<|image|>"]],  # high-quality synthetic  superior
                    max_gen_len=10240,
                    temperature=t,
                    logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                )
            generated[1][0].save(args.save_path + f"{args.i2i_task}.png")
        else:
            for i, prompt in enumerate(l_prompts):
                for repeat_idx in range(n):
                    random_seed(repeat_idx)
                    if args.task == 't2i':
                        generated = inference_solver.generate(
                                images=[],
                                qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{prompt}", None]],  # high-quality synthetic  superior
                                max_gen_len=10240,
                                temperature=t,
                                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                            )
                    else:
                        task_dict = {"depth": "depth map", "canny": "canny edge map", "hed": "hed edge map", "openpose":"pose estimation map"}
                        generated = inference_solver.generate(
                                images=[],
                                qas=[[f"Generate a dual-panel image of {w}x{h} where the <lower half> displays a <{task_dict[args.task]}>, while the <upper half> retains the original image for direct visual comparison:\n{prompt}" , None]], 
                                max_gen_len=10240,
                                temperature=t,
                                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                                )
                    generated[1][0].save(args.save_path + f"{i}_{repeat_idx}.png")
