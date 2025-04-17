from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import json
from PIL import Image
from io import BytesIO
import base64
import argparse
from src.scripts.run_model import main


app = FastAPI()


class InputData(BaseModel):
    message: str | None = None
    file: List | None = None

def convert_str_to_array(message: str):
    # Tìm vị trí đầu tiên của ký tự '['
    start_index = message.find('[')

    # Tìm vị trí cuối cùng của ký tự ']'
    end_index = message.rfind(']')
    # print(message[start_index:end_index+1])
    return json.loads(message[start_index:end_index+1])

def check_exists_objects(list_objects, vocab):

    if len(list_objects) == 0 or len(set(list_objects) - set(vocab)) != 0:
        return False
    else:
        return True
    
def check_exists_relations(list_relations, vocab):

    relations = [rel[1] for rel in list_relations]

    if len(set(relations) - set(vocab)) != 0:
        return False
    else:
        return True
    
def image_array_to_base64(image_array):
    image = Image.fromarray(image_array.astype('uint8'))  # tạo ảnh từ numpy array
    buffered = BytesIO()  # tạo một buffer trong RAM (không phải file trên ổ đĩa)
    image.save(buffered, format="PNG")  # lưu ảnh vào buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # mã hóa ảnh sang base64

def convert_sentence(message: str):

    # Set up your OpenAI API key
    client = OpenAI(api_key="<OPENAI_KEY>")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that identifies entities and the relationships between them in a sentence, and converts the information into a specific format."
                    """Example:
                            Sentence: The sky is above the grass, a sheep is standing on the grass, another sheep is by the first sheep, and a tree is behind the first sheep.
                            Format Conversion:
                                [
                                    {
                                        "objects": ["sky", "grass", "sheep", "sheep", "tree"],
                                        "relationships": [
                                            [0, "above", 1],
                                            [2, "standing on", 1],
                                            [3, "by", 2],
                                            [4, "behind", 2]
                                        ]
                                    }
                                ]    
                    """
                )
            },
            {
                "role": "user",
                "content": (
                    f"Identify the objects and find the relationships between them in the sentence {message}. Then convert them into the appropriate format."
                )
            },
        ]
    )
    return response.choices[0].message.content


def other_question(message: str):

    # Set up your OpenAI API key
    client = OpenAI(api_key="<OPENAI_KEY>")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that answer the question."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Answer the question: {message}."
                )
            },
        ]
    )
    return response.choices[0].message.content


def run_image_generation(checkpoint, scene_graphs_json, output_dir, scene_graphs, draw_scene_graphs=0, device='gpu'):
    args = argparse.Namespace(
        checkpoint=checkpoint,
        scene_graphs_json=scene_graphs_json,
        output_dir=output_dir,
        draw_scene_graphs=draw_scene_graphs,
        device=device
    )

    output = main(args, scene_graphs)
    return output


@app.post("/image-generation")
async def image_generation(data: InputData):

    objects = ['window', 'tree', 'man', 'shirt', 'wall', 'person', 'building', 'ground', 
               'sign', 'light', 'sky', 'head', 'leaf', 'leg', 'hand', 'pole', 'grass', 
               'hair', 'car', 'woman', 'cloud', 'ear', 'eye', 'line', 'table', 'shoe', 
               'people', 'door', 'shadow', 'wheel', 'letter', 'pant', 'flower', 'water', 
               'chair', 'fence', 'floor', 'handle', 'nose', 'arm', 'plate', 'stripe', 
               'rock', 'jacket', 'hat', 'tail', 'foot', 'face', 'road', 'tile', 'number', 
               'sidewalk', 'short', 'spot', 'bag', 'snow', 'bush', 'boy', 'helmet', 'street', 
               'field', 'bottle', 'glass', 'tire', 'logo', 'background', 'roof', 'post', 
               'branch', 'boat', 'plant', 'umbrella', 'brick', 'picture', 'girl', 'button', 
               'mouth', 'track', 'part', 'bird', 'food', 'box', 'banana', 'dirt', 'cap', 'jean', 
               'glasses', 'bench', 'mirror', 'book', 'pillow', 'top', 'wave', 'shelf', 'clock', 
               'glove', 'headlight', 'bowl', 'trunk', 'bus', 'neck', 'edge', 'train', 'reflection', 
               'horse', 'paper', 'writing', 'kite', 'flag', 'seat', 'house', 'wing', 'board', 
               'lamp', 'cup', 'elephant', 'cabinet', 'coat', 'mountain', 'giraffe', 'sock', 
               'cow', 'counter', 'hill', 'word', 'finger', 'dog', 'wire', 'sheep', 'zebra', 
               'ski', 'ball', 'frame', 'back', 'bike', 'truck', 'animal', 'design', 'ceiling', 
               'sunglass', 'sand', 'skateboard', 'motorcycle', 'curtain', 'container', 'windshield', 
               'cat', 'towel', 'beach', 'knob', 'boot', 'bed', 'sink', 'paw', 'surfboard', 'horn', 
               'pizza', 'wood', 'bear', 'stone', 'orange', 'engine', 'photo', 'hole', 'child', 
               'railing', 'player', 'stand', 'ocean', 'lady', 'vehicle', 'sticker', 'pot', 
               'apple', 'basket', 'plane', 'key', 'tie']
    
    relations = ['has', 'covering', 'next to', 'above', 'belonging to', 'and', 'have', 'beside', 
                 'behind', 'by', 'laying on', 'hanging on', 'eating', 'under', 'for', 'on side of', 
                 'standing in', 'with', 'standing on', 'below', 'of', 'against', 'attached to', 
                 'parked on', 'holding', 'on top of', 'carrying', 'at', 'on', 'wearing', 
                 'in front of', 'looking at', 'wears', 'sitting in', 'near', 'over', 'sitting on', 
                 'in', 'inside', 'walking on', 'along', 'made of', 'riding', 'covered in', 'around']


    if data.file is None:
        scene_graph = convert_str_to_array(convert_sentence(data.message))
        scene_graph_objs = scene_graph[0]['objects']
        scene_graph_rels = scene_graph[0]['relationships']

        if check_exists_objects(scene_graph_objs, objects) and check_exists_relations(scene_graph_rels, relations):
            image = run_image_generation(
                checkpoint="src/sg2im-models/vg128.pt",
                scene_graphs_json="src/scene_graphs/figure_6_sheep.json",
                output_dir="outputs",
                scene_graphs=scene_graph,
                draw_scene_graphs=0,
                device="gpu"
            )

            return {"message": "success", "image": image_array_to_base64(image)}
        
        else:
            response = other_question(data.message)
            return {"message": "failed", "image": response}
        
    else:
        image = run_image_generation(
            checkpoint="src/sg2im-models/vg128.pt",
            scene_graphs_json="src/scene_graphs/figure_6_sheep.json",
            output_dir="outputs",
            scene_graphs=data.file,
            draw_scene_graphs=0,
            device="gpu"
        )

        return {"message": "success", "image": image_array_to_base64(image)}

    
# Cho phép gọi từ frontend (JavaScript)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5501"],  # hoặc chỉ domain bạn muốn, ví dụ: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
