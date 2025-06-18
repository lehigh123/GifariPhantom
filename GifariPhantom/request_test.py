import requests
import json
import base64
from datetime import datetime
import re
import subprocess

url = "https://api.cortex.cerebrium.ai/v4/p-70eb8ec7/gifariphantom/run"

# Generate a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a safe filename from the prompt
# prompt = "The warm sun shines over the grass. A little girl with twin ponytails, a green bow on her head, and a light green dress squats next to the blooming daisies. Next to her, a brown and white dog sticks out its tongue, its furry tail wagging happily. The little girl smiles and holds up a yellow and red toy camera with a blue button to freeze the happy moment with the dog."
# prompt = "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。"
prompt = "An older woman takes out her yellow toy camera and photographs the grand canyon"
# Take first 30 chars of prompt, remove special chars, replace spaces with underscores
safe_prompt = (
    re.sub(r"[^a-zA-Z0-9\s]", "", prompt[:30])
    .strip()
    .replace(" ", "_")
    .lower()
)

# Create unique filename
filename = f"s2v_720x1280_1_1_{safe_prompt}_{timestamp}.mp4"

payload = json.dumps(
    {
        "input": {
            "task": "s2v-1.3B",
            "size": "832*480",
            "frame_num": 81,
            "sample_fps": 10,
            "ckpt_dir": "/persistent-storage/Wan2.1-T2V-1.3B",
            "phantom_ckpt": "/persistent-storage/Phantom-Wan-Models/Phantom-Wan-1.3B.pth",
            "offload_model": None,
            "ulysses_size": 1,
            "ring_size": 1,
            "t5_fsdp": False,
            "t5_cpu": False,
            "dit_fsdp": False,
            "save_file": f"/persistent-storage/{filename}",
            "prompt": prompt,
            "use_prompt_extend": False,
            "prompt_extend_method": "local_qwen",
            "prompt_extend_model": None,
            "prompt_extend_target_lang": "ch",
            "base_seed": 42,
            "image": None,
            "ref_image": "/persistent-storage/Phantom-Wan-Models/assets/ref1.png,/persistent-storage/Phantom-Wan-Models/assets/ref6.png",
            "sample_solver": "unipc",
            "sample_steps": 50,
            "sample_shift": 5.0,
            "sample_guide_scale": 5.0,
            "sample_guide_scale_img": 5.0,
            "sample_guide_scale_text": 7.5,
        }
    }
)

headers = {
    "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTcwZWI4ZWM3IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY1NTY4OTAwfQ.o_ANkVWZIJvuCYvzqgCV8p78zMw5A2AyPxJQF9lpc6IUJaFFAOvY2otegcWohIDbm6k5EfvfteJWrRISifYyFDA2xbJvfwCPxiEp3MNSssiGRcMPOkxxXPZ3foBeOUBn2re_EVSdhp49AKKDwu3v47aGbtCdOUiagb5sdOBnh8S40ZCqVqaaMyQ5Yqmd3aJn_ipVzWvcGeVj69eB0IOi6jCDxRLjSh5Ej7UDrKR6sc77eBW1wEz8JV9Z5NA2USPsH8GcJcY3cGo8B0eTdY54GCACoyMTh1YHq_gwmwlcciVfHtnCf3_36mPORab_FsHU9wKR0dET6i5mwjwzOBe0kg",
    "Content-Type": "application/json",
}

# Store the file path globally
file_path = None

try:
    # Set a longer timeout (20 minutes) and stream the response
    response = requests.post(
        url,
        headers=headers,
        data=payload,
        timeout=1200,  # 20 minutes timeout
        stream=True,
    )

    print(f"Status Code: {response.status_code}")
    print("Response Headers:", response.headers)

    if response.status_code == 200:
        try:
            # Read the response content
            response_text = response.text
            print("\nRaw Response:", response_text)

            # Try to parse the JSON response
            response_data = json.loads(response_text)
            print(
                "\nParsed JSON Response:", json.dumps(response_data, indent=2)
            )

            # Check if we got a file path in the response
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
if file_path:
    # Extract just the filename from the full path
    file_name = file_path.split('/')[-1]
    print(f"\nDownloading file: {file_name}")
    
    # Download the file using cerebrium command
    try:
        subprocess.run(['cerebrium', 'download', file_name], check=True)
        print(f"Successfully downloaded {file_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {str(e)}")
    except FileNotFoundError:
        print("Error: 'cerebrium' command not found. Please make sure cerebrium CLI is installed.")

# Parse the JSON response
response_data = json.loads(response.text)

# Extract the base64 image data
image_data = response_data.get("result", "").get(
    "image", ""
)  # Adjust this key based on your API response structure

# Decode base64 and save to file
if image_data:
    image_bytes = base64.b64decode(image_data)
    with open("output_image.png", "wb") as f:
        f.write(image_bytes)
    print("Image saved as 'output_image.png'")
else:
    print("No image data found in response")
