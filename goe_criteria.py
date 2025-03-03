import os
import pickle

jump_criteria_positive=["unexpected or creative or difficult entry",
                        "containing clear recognizable (difficult for jump preceded by steps/movements of the Short Program) steps/free skating movements immediately preceding element",
                        "containing varied position in the air / delay in rotation",
                        "containing good height and distance",
                        "containing good extension on landing / creative exit",
                        "containing good flow from entry to exit including jump combinations / sequences",
                        "effortless throughout",
                        "matched to the musical structure"
                        ]
jump_criteria_positive = [f"a photo of a jump that is {jump_criteria}" for jump_criteria in jump_criteria_positive]
spin_criteria_positive=["containing good speed or acceleration during spin",
                        "centering a spin quickly",
                        "containing balanced rotations in all positions",
                        "clearly more than required number of revolutions",
                        "containing good, strong position(s) (including height and air/landing position in flying spins)",
                        "creative and original",
                        "good control throughout all phases",
                        "matched to the musical structure"
                        ]
spin_criteria_positive = [f"a photo of a spin that is {spin_criteria}" for spin_criteria in spin_criteria_positive]

step_seq_criteria_positive=["containing good energy and execution",
                        "containing good speed or acceleration during sequence",
                        "containing the use of well executed various steps during the sequence",
                        "containing deep clean edges (including entry and exit of all turns)",
                        "indicating good control and commitment of the whole body maintaining accuracy of steps",
                        "creative and original",
                        "effortless throughout",
                        "enhances the musical structure"
                        ]
step_seq_criteria_positive = [f"a photo of a step sequence that is {step_seq_criteria}" for step_seq_criteria in step_seq_criteria_positive]

choreographic_seq_criteria_positive=["containing good flow, energy and execution",
                        "containing good speed or acceleration during sequence",
                        "containing good clarity and precision",
                        "containing good control and commitment of whole body",
                        "is creative and original",
                        "effortless throughout",
                        "reflecting concept or character of the program",
                        "enhancing the musical structure"
                        ]
choreographic_seq_criteria_positive = [f"a photo of a choreographic step sequence that is or has {choreographic_seq_criteria}" for choreographic_seq_criteria in choreographic_seq_criteria_positive]
judging_reductions = {
    "jump where reduction is due to element ": {
        "Combo of one jump final GOE must be": [-3],
        "Downgraded (sign << )": [-2, -3],
        "No required preceding steps/movements": [-3],
        "Under-rotated (sign < )": [-1, -2],
        "Break between required steps/movements & jump/only 1 step/movement preceding jump": [-1, -2],
        "Lacking rotation (no sign) including half loop in a combo": [-1],
        "Fall": [-3],
        "Poor speed, height, distance, air position": [-1, -2],
        "Landing on two feet in a jump": [-3],
        "Touch down with both hands in a jump": [-2],
        "Stepping out of landing in a jump": [-2, -3],
        "Touch down with one hand or free foot": [-1],
        "2 three turns in between (jump combo)": [-2],
        "Loss of flow/direction/rhythm between jumps (combo/seq.)": [-1, -2],
        "Severe wrong edge take off F/Lz (sign “e”)": [-2, -3],
        "Weak landing (bad pos./wrong edge/scratching etc)": [-1, -2],
        "Unclear or wrong edge take off F/Lz (sign “!” or no sign)": [-1, -2],
        "Poor take-off": [-1, -2],
        "Long preparation": [-1, -2],
    },
    "spin where reduction is due to element ": {
        "Prescribed air position not attained (flying spin)": [-1, -2],
        "Poor/awkward, unaesthetic position(s)": [-1, -3],
        "Fall": [-3],
        "Traveling": [-1, -3],
        "Touch down with both hands": [-2],
        "Slow or reduction of speed": [-1, -3],
        "Touch down with free foot or one hand": [-1, -2],
        "Change of foot poorly done (including curve of entry/exit except when changing direction)": [-1, -3],
        "Less than required revolutions": [-1, -2],
        "Incorrect take-off or landing in a flying spin": [-1, -2],
        "Poor fly (flying spin/entry)": [-1, -3],
    },
    "step sequence where reduction is due to element ": {
        "Listed jumps with more than half revolution included": [-1],
        "Poor quality of steps, turns, positions": [-1, -3],
        "Fall": [-3],
        "Stumble": [-1, -2],
        "Less than half of the pattern doing steps/turns": [-2, -3],
        "Does not correspond to the music": [-1, -2],
    },
    "choreographic sequence where reduction is due to element ": {
        "Fall": [-3],
        "Stumble": [-1, -2],
        "Inability to clearly demonstrate the sequence": [-2, -3],
        "Does not enhance the music": [-1, -3],
        "Loss of control while executing the sequence": [-1, -3],
        "Poor quality of movements": [-1, -2],
    },
}

def get_all_positive_criteria_in_one():
    all_positive_criteria = []
    all_positive_criteria.extend(jump_criteria_positive)
    all_positive_criteria.extend(spin_criteria_positive)
    all_positive_criteria.extend(step_seq_criteria_positive)
    all_positive_criteria.extend(choreographic_seq_criteria_positive)
    return all_positive_criteria

def get_all_negative_criteria_in_one():
    weights = []
    text_prompts = []
    for element_type in judging_reductions:
        for reduction_desc, reduction_value in judging_reductions[element_type].items():
            text_prompt = f"{element_type} {reduction_desc}"
            text_prompts.append(text_prompt.lower())
            if len(reduction_value) > 1:
                reduction_value=sum(reduction_value)//2
            else:
                reduction_value=reduction_value[0]
            weights.append(reduction_value)
    return text_prompts, weights

if __name__ == "__main__":
    import torch
    positive_text_prompts = get_all_positive_criteria_in_one()
    positive_weights = [0.5 for item in positive_text_prompts]
    negative_text_prompts, negative_weights = get_all_negative_criteria_in_one()
    print(len(positive_text_prompts))
    print(len(negative_text_prompts))
    from collections import defaultdict
    lookup_dict = {"positive": defaultdict(list), "negative": defaultdict(list)}
    for text_prompt in positive_text_prompts:
        lookup_dict['positive']['jump'].append(int('jump' in text_prompt))
        lookup_dict['positive']['choreographic_ss'].append(int('a photo of a choreographic step sequence' in text_prompt))
        lookup_dict['positive']['ss'].append(int('a photo of a step sequence' in text_prompt))
        lookup_dict['positive']['spin'].append(int('spin' in text_prompt))
    for text_prompt in negative_text_prompts:
        lookup_dict['negative']['jump'].append(int('jump where reduction' in text_prompt))
        lookup_dict['negative']['choreographic_ss'].append(int('choreographic sequence where reduction' in text_prompt))
        lookup_dict['negative']['ss'].append(int('step sequence where reduction' in text_prompt))
        lookup_dict['negative']['spin'].append(int('spin where reduction' in text_prompt))
    print([sum(lookup_dict['positive'][key]) for key in lookup_dict['positive'] ])
    print([sum(lookup_dict['negative'][key]) for key in lookup_dict['negative'] ])
    with open("super_action_mask_lookup.pkl", 'wb') as f:
        pickle.dump(lookup_dict, f)
    exit()

    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)


    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")#.to(4)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True).to(4)


    # Tokenize and encode the prompts
    positive_inputs = processor(positive_text_prompts,images=image, return_tensors="pt", padding=True, truncation=True)
    negative_inputs = processor(negative_text_prompts,images=image, return_tensors="pt", padding=True, truncation=True)

    # Get the CLIP text embeddings
    with torch.no_grad():
        pos_outputs = model(**positive_inputs)
        neg_outputs = model(**negative_inputs)

    # Extract the text embeddings from the output
    pos_text_embeddings = pos_outputs['text_model_output']['pooler_output']
    neg_text_embeddings = neg_outputs['text_model_output']['pooler_output']
    print("Positive embeddings shape:", pos_text_embeddings.shape)
    print("Negative embeddings shape:", neg_text_embeddings.shape)
    save_path = "pooler_text_embeddings_and_weights.pth"
    torch.save({"positive_text_embeddings": pos_text_embeddings, "negative_text_embeddings": neg_text_embeddings, "pos_weights": positive_weights, "neg_weights": negative_weights}, save_path)



