skill_specific_deductions = [
    ('loss and retrieval of apparatus without traveling', [-0.5]),
    ('loss and retrieval of apparatus after 1–2 steps', [-0.7]),
    ('loss and retrieval of apparatus after 3 or more steps', [-1.0]),
    ('loss of apparatus outside the floor area', [-1.0]),
    ('throw with imprecise trajectory or incorrect direction', [-0.1, -0.5]),
    ('deviation in apparatus flight during throw', [-0.1, -0.3]),
    ('involuntary catch or grip of apparatus', [-0.1, -0.3]),
    ('incorrect handling of apparatus, such as ball held against forearm', [-0.1, -0.3]),
    ('incomplete rotation of apparatus during an element', [-0.1, -0.3]),
    ('loss of balance without fall during body movement', [-0.3, -0.5]),
    ('fall during body movement', [-0.7]),
    ('incorrect body posture during elements', [-0.1, -0.3]),
    ('loss of flow during routine', [-0.1, -0.3]),
    ('inability to maintain control of apparatus', [-0.1, -0.3]),
    ('poor quality of steps or turns in choreography', [-0.1, -0.3]),
    ('stumble during choreography', [-0.1, -0.2]),
    ('fall during choreography', [-0.7]),
    ('loss of control in executing elements', [-0.1, -0.3]),
    ('failure to execute required number of rotations in a spin', [-0.1, -0.3]),
    ('spin with traveling or loss of position', [-0.1, -0.3]),
    ('touch down with hands or feet during a spin', [-0.1, -0.3]),
    ('poor aesthetic positions during spins or body movements', [-0.1, -0.3])
]

performance_wide_deductions = [
    ('incomplete synchronization with music', [-0.3]),
    ('absence of harmony between movements and music', [-0.5]),
    ('lack of variety in movements', [-0.2]),
    ('insufficient use of the floor area', [-0.1]),
    ('lack of facial expression or projection', [-0.1]),
    ('boundary violation (gymnast or apparatus leaving floor area)', [-0.3]),
    ('time infraction (routine exceeds or falls short of allowed duration)', [-0.05]),
    ('choreography does not enhance the music', [-0.1, -0.3]),
    ('poor quality of movements in choreography', [-0.1, -0.2])
]


def get_all_positive_criteria_in_one():
    # Placeholder for rhythmic gymnastics as positive criteria are subjective and not deduction-based
    return ["placeholder"], [0]

def get_all_negative_criteria_in_one():
    weights = []
    text_prompts = []
    for reduction_desc, reduction_value in skill_specific_deductions:
        text_prompt = reduction_desc
        text_prompts.append(text_prompt.lower())
        if len(reduction_value) > 1:
            reduction_value = sum(reduction_value) / len(reduction_value)
        else:
            reduction_value = reduction_value[0]
        weights.append(reduction_value)
    return text_prompts, weights
