### Useful commands:

#### Cerebrium
* ```cerebrium deploy```
* ```curl --request GET --url https://rest.cerebrium.ai/v2/hardware | jq```

#### Test
* Run unit tests: ```uv run pytest tests/```

#### Send requests:
* Subject to video 

```
uv run request_test.py --frame_num 10 \
    --ref_images /Users/doms/my_ref_images/tiny_portrait_1.jpeg,/Users/doms/my_ref_images/charlie_portrait.jpeg  \
    --size "832*480" \
    --sampling_steps 30 \
    --prompt "A black and gray cat smoking marijuana and blowing smoke in a cartoon" \
    --useWebhook "https://d087-2601-47-477e-1a60-3945-2ba-21e7-583a.ngrok-free.app/webhook" \
    --useAsync
```

```
uv run request_test.py --frame_num 10 \
    --ref_images /Users/doms/my_ref_images/tiny_portrait_1.jpeg,/Users/doms/my_ref_images/charlie_portrait.jpeg  \
    --size "832*480" \
    --sampling_steps 30 \
    --prompt "A black and gray cat smoking marijuana and blowing smoke in a cartoon" \
    --useWebhook "https://d087-2601-47-477e-1a60-3945-2ba-21e7-583a.ngrok-free.app/webhook" \
    --useAsync \
    --use_prompt_extend
```

* ngrok: [http://127.0.0.1:4040/status](http://127.0.0.1:4040/status)

#### Run local webhook:
* ```uv run webhook.py```


#### Run remote webhook:
* ```ngrok http 5000```