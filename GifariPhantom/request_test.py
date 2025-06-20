import requests
import json
import base64
from datetime import datetime
import re
import subprocess
import argparse
import logging
from src.image_utils.image_utils import image_path_to_base64
import copy

# url = "https://api.cortex.cerebrium.ai/v4/p-70eb8ec7/gifariphantom/run"
# url = "https://api.cortex.cerebrium.ai/v4/p-70eb8ec7/phantom/run"
url = "https://api.cortex.cerebrium.ai/v4/p-a2961cf2/phantom/run"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    default_prompts = [
        "The warm sun shines over the grass. A little girl with twin ponytails, a green bow on her head, and a light green dress squats next to the blooming daisies. Next to her, a brown and white dog sticks out its tongue, its furry tail wagging happily. The little girl smiles and holds up a yellow and red toy camera with a blue button to freeze the happy moment with the dog.",
        "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。",
        "An older woman takes out her yellow toy camera and photographs the grand canyon"
    ]

    parser = argparse.ArgumentParser(description="Run GifariPhantom API request.")
    parser.add_argument('--ref_images', type=str, required=True, help='Comma-separated reference image paths')
    parser.add_argument('--prompt', type=str, default=default_prompts[1], help='Prompt for the video generation. Defaults to a sample prompt. Available test prompts:\n1. {}\n2. {}\n3. {}'.format(*default_prompts))
    parser.add_argument('--frame_num', type=int, default=81, help='Number of frames')
    parser.add_argument('--size', type=str, default="832*480", help='Video size, e.g., 832*480')
    parser.add_argument('--useWebhook', type=str, default=None, help='Optional webhook endpoint URL')
    parser.add_argument('--useAsync', action='store_true', help='If set, appends async=true to the API URL')
    parser.add_argument('--sampling_steps', type=int, default=50, help='Number of sampling steps (default: 50)')
    parser.add_argument('--base_seed', type=int, default=42, help='Base seed for video generation (default: 42)')
    parser.add_argument('--task', type=str, default="s2v-1.3B", help='Task type for video generation (default: s2v-1.3B)')
    parser.add_argument('--phantom_ckpt', type=str, default="/persistent-storage/Phantom-Wan-Models/Phantom-Wan-1.3B.pth", help='Path to Phantom-Wan checkpoint (default: /persistent-storage/Phantom-Wan-Models/Phantom-Wan-1.3B.pth)')
    parser.add_argument('--offload_model', action='store_true', help='Offload model to CPU after each forward pass to save GPU memory')
    parser.add_argument('--t5_cpu', action='store_true', help='Place T5 model on CPU to save GPU memory')
    parser.add_argument('--t5_fsdp', action='store_true', help='Use FSDP (Fully Sharded Data Parallel) for T5 model')
    parser.add_argument('--dit_fsdp', action='store_true', help='Use FSDP for DiT model')
    parser.add_argument('--ulysses_size', type=int, default=1, help='Size of ulysses parallelism in DiT (default: 1)')
    parser.add_argument('--ring_size', type=int, default=1, help='Size of ring attention parallelism in DiT (default: 1)')
    parser.add_argument('--sample_shift', type=float, default=5.0, help='Sampling shift factor for flow matching schedulers (default: 5.0)')
    parser.add_argument('--use_prompt_extend', action='store_true', help='Enable prompt extension to enhance input prompts')
    parser.add_argument('--prompt_extend_method', type=str, choices=['dashscope', 'local_qwen'], default='local_qwen', help='Prompt extension method (default: local_qwen)')
    parser.add_argument('--prompt_extend_model', type=str, default=None, help='Model name for prompt extension (default: auto-selected based on method)')
    parser.add_argument('--prompt_extend_target_lang', type=str, choices=['ch', 'en'], default='en', help='Target language for prompt extension (default: en)')
    parser.add_argument('--hf_cache_dir', type=str, default='/persistent-storage/hf_cache', help='Hugging Face cache directory (default: /persistent-storage/hf_cache)')
    args = parser.parse_args()

    # Set the API URL, append webhookEndpoint and async if provided
    api_url = url
    query_params = []
    if args.useWebhook:
        query_params.append(f"webhookEndpoint={args.useWebhook}")
    if args.useAsync:
        query_params.append("async=true")
    if query_params:
        sep = '&' if '?' in api_url else '?'
        api_url = f"{api_url}{sep}{'&'.join(query_params)}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = (
        re.sub(r"[^a-zA-Z0-9\s]", "", args.prompt[:30])
        .strip()
        .replace(" ", "_")
        .lower()
    )
    filename = f"{args.task}_{args.size.replace('*', 'x')}_{args.frame_num}_{safe_prompt}_{timestamp}.mp4"

    # Convert ref_images to base64
    ref_image_paths = [img.strip() for img in args.ref_images.split(",") if img.strip()]
    ref_images_base64 = [image_path_to_base64(path) for path in ref_image_paths]
    ref_images_base64_str = ",".join(ref_images_base64)

    payload_dict = {
        "input": {
            "task": args.task,
            "size": f"{args.size}",
            "frame_num": args.frame_num,
            "sample_fps": 10,
            "ckpt_dir": "/persistent-storage/Wan2.1-T2V-1.3B",
            "phantom_ckpt": args.phantom_ckpt,
            "offload_model": args.offload_model,
            "ulysses_size": args.ulysses_size,
            "ring_size": args.ring_size,
            "t5_fsdp": args.t5_fsdp,
            "t5_cpu": args.t5_cpu,
            "dit_fsdp": args.dit_fsdp,
            "save_file": f"/persistent-storage/{filename}",
            "prompt": args.prompt,
            "use_prompt_extend": args.use_prompt_extend,
            "prompt_extend_method": args.prompt_extend_method,
            "prompt_extend_model": args.prompt_extend_model,
            "prompt_extend_target_lang": args.prompt_extend_target_lang,
            "base_seed": args.base_seed,
            "image": None,
            "ref_image": ref_images_base64_str,
            "sample_solver": "unipc",
            "sample_steps": args.sampling_steps,
            "sample_shift": args.sample_shift,
            "sample_guide_scale": 5.0,
            "sample_guide_scale_img": 5.0,
            "sample_guide_scale_text": 7.5,
        }
    }
    payload = json.dumps(payload_dict)

    # 1. Log the input request (pretty formatted, redact base64 ref_image)
    log_payload = copy.deepcopy(payload_dict)
    if "input" in log_payload and "ref_image" in log_payload["input"]:
        ref_imgs = log_payload["input"]["ref_image"].split(",")
        log_payload["input"]["ref_image"] = f"<{len(ref_imgs)} base64 images, lengths: {[len(s) for s in ref_imgs]}>"
    logging.info("Input request payload:\n%s", json.dumps(log_payload, indent=2, ensure_ascii=False))
    # 2. Log the output file name
    logging.info("Output file name: %s", filename)

    headers = {
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWEyOTYxY2YyIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY1OTEwNDk3fQ.cSJS--BJ2ThZqzwROS9jf00gOFQ57iTN-OkZxlHNE5w1HLjNpH0OZpg2xjYzxcA05vngjoIqSPDo7paUi34JwiH0Azh1upurkWrRNHevDasd1movEwbNdAH0yY533sZR4t-q_bfZpgFhS3ltZDeNYiogwmhDdME0pFTZE-FPYpRk2VVTwKI_Tr23frdU-OaoN8AFtc89C_cISoEgKV3BfM47hipTCp5BazMHIqv8SyAh2pAgVGiKIzXddKO1fhm5ZSZnmvBVIYcG6sCMcBevaBqvcZAORpJTatj3kEP2b2840GRsE6xpeaRqDW9IBOG_0fmVShuPcOzZxfZzkGAsYA",
        "Content-Type": "application/json",
    }

    file_path = None
    run_id = None

    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=payload,
            timeout=1200,  # 20 minutes timeout
            stream=True,
        )

        # 3. Log the response status code
        logging.info("Response status code: %s", response.status_code)
        print(f"Status Code: {response.status_code}")
        print("Response Headers:", response.headers)

        if response.status_code == 200 or response.status_code == 202:
            try:
                response_text = response.text
                print("\nRaw Response:", response_text)
                response_data = json.loads(response_text)
                # 4. Log the pretty-printed response
                logging.info("Response JSON (pretty):\n%s", json.dumps(response_data, indent=2, ensure_ascii=False))
                print("\nParsed JSON Response:", json.dumps(response_data, indent=2))
                # Handle async response
                if args.useAsync and 'run_id' in response_data:
                    run_id = response_data['run_id']
                    print(f"\nAsync run started. Run ID: {run_id}")
                    logging.info("Async run started. Run ID: %s", run_id)
                    return  # Skip file download/image extraction
                if "result" in response_data and "result" in response_data["result"] and "file" in response_data["result"]["result"]:
                    print("\nVideo generation completed successfully!")
                    file_path = response_data['result']['result']['file']
                    print(f"Video saved to: {file_path}")
                else:
                    print("\nWarning: No file path found in response")
            except json.JSONDecodeError as e:
                print("\nError parsing JSON response:", str(e))
                print("Raw response content:", response_text)
        else:
            print(f"\nRequest failed with status code: {response.status_code}")
            print("Response content:", response.text)

    except requests.exceptions.Timeout:
        print(
            "\nRequest timed out after 20 minutes. The video generation might still be running on the server."
        )
        print(
            "Check the server logs for progress and the output directory for the generated video."
        )
    except requests.exceptions.RequestException as e:
        print("\nRequest failed:", str(e))

    # Download the file if we have a valid file path
    if not args.useAsync and file_path:
        file_name = file_path.split('/')[-1]
        logging.info("Cerebrium download command: cerebrium download %s", file_name)
        print(f"\nDownloading file: {file_name}")
        try:
            subprocess.run(['cerebrium', 'download', file_name], check=True)
            logging.info("Successfully downloaded %s", file_name)
            print(f"Successfully downloaded {file_name}")
        except subprocess.CalledProcessError as e:
            logging.error("Error downloading file: %s", str(e))
            print(f"Error downloading file: {str(e)}")
        except FileNotFoundError:
            logging.error("Error: 'cerebrium' command not found. Please make sure cerebrium CLI is installed.")
            print("Error: 'cerebrium' command not found. Please make sure cerebrium CLI is installed.")

    # Parse the JSON response
    if not args.useAsync:
        try:
            response_data = json.loads(response.text)
        except Exception:
            response_data = {}

        image_data = response_data.get("result", "").get(
            "image", ""
        )  # Adjust this key based on your API response structure

        if image_data:
            image_bytes = base64.b64decode(image_data)
            with open("output_image.png", "wb") as f:
                f.write(image_bytes)
            print("Image saved as 'output_image.png'")
        else:
            print("No image data found in response")

if __name__ == "__main__":
    main()
