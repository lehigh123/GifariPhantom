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
    --useWebhook "https://8593-2601-87-8300-4110-c980-362-ca7e-d436.ngrok-free.app/webhook" \
    --useAsync
```

```
uv run request_test.py --frame_num 75 \
    --ref_images /Users/doms/my_ref_images/59817659-C5CC-4FEB-9DCE-03DC7F69523A.png,/Users/doms/my_ref_images/tong.png \
    --size "832*480" \
    --sampling_steps 50 \
    --prompt "Cartoon style: A cat and a baseball bat cartoon character are lounging on the beach" \
    --useWebhook "https://8593-2601-87-8300-4110-c980-362-ca7e-d436.ngrok-free.app/webhook" \
    --useAsync \
    --base_seed -1
```

* really short request:
```
uv run request_test.py --frame_num 1 \
    --ref_images /Users/doms/my_ref_images/59817659-C5CC-4FEB-9DCE-03DC7F69523A.png,/Users/doms/my_ref_images/tong.png \
    --size "832*480" \
    --sampling_steps 50 \
    --prompt "Cartoon style: A cat and a baseball bat cartoon character are lounging on the beach" \
    --base_seed -1
```

* reference request:
** 1.3B
```
uv run request_test.py --ref_images tests/test-data/ref1.png,tests/test-data/ref2.png --size "832*480" 

```


** 14B
```
uv run request_test.py --ref_images "tests/test-data/ref12.png,tests/test-data/ref13.png" --size "832*480" \
    --task i2v-14B  \
    --frame_num 81 \
    --prompt "扎着双丸子头，身着红黑配色并带有火焰纹饰服饰，颈戴金项圈、臂缠金护腕的哪吒，和有着一头淡蓝色头发，额间有蓝色印记，身着一袭白色长袍的敖丙，并肩坐在教室的座位上，他们专注地讨论着书本内容。背景为柔和的灯光和窗外微风拂过的树叶，营造出安静又充满活力的学习氛围。"
```
* ngrok: [http://127.0.0.1:4040/status](http://127.0.0.1:4040/status)

#### Run local webhook:
* ```uv run webhook.py```


#### Run remote webhook:
* ```ngrok http 5000```