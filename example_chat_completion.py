# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

#     dialogs: List[Dialog] = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#     ]

    dialogs = [
        [
            {
                "role": "user",
                "content": """Get product name, size, color, address, phone number from this conversation:
```
e ơi
Chào chị, em có thể giúp chị với điều gì ạ?
e còn áo thun nào ko, chị đang cần mua gấp
Dạ chị ơi, bên em có nhiều mẫu áo thun đẹp lắm ạ. Chị có yêu cầu cụ thể nào về chất liệu hoặc kiểu dáng không để em gợi ý cho chị ạ?
áo thun mặc bên trong áo khoác cho ấm ấy
Dạ chị ơi, bên em có Áo Nỉ Kéo Khóa rất phù hợp để mặc bên trong áo khoác cho ấm ạ. Áo này có chất cotton lót lông thoáng mát và cổ tay áo được sử dụng vải gân giúp giữ phom. Size chị mặc là bao nhiêu ạ?
chị 45kg thì nên mặc size nào
Dựa vào cân nặng của chị, chị có thể chọn size XS hoặc S tùy theo sở thích về ôm hay rộng hơn ạ. Nếu chị thích áo ôm sát thì chị có thể chọn size XS, còn nếu thích rộng rãi hơn thì chị có thể chọn size S ạ.
thế cho chị lấy size S, có màu gì nhỉ
Sản phẩm Áo Nỉ Kéo Khóa size S có các màu: vàng, đen, nâu và trắng ạ. Chị muốn chọn màu nào?
màu nào mặc lên trông gầy hơn hả em, chị hơi nhiều mỡ
Chị có thể chọn màu đen hoặc màu nâu ạ, vì hai màu này sẽ giúp tạo cảm giác thon gọn hơn ạ.
thế chị lấy màu đen nhé
Dạ vâng, em sẽ ghi chú lại là chị chọn size S màu đen ạ. Chị vui lòng cho em xin số điện thoại và địa chỉ nhận hàng để em tạo đơn hàng cho chị ạ.
0389266155 s105 smart city
```
"""
            }
        ]
    ]

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
