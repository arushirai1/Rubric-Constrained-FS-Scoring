# __all__ = ['hand_negatives_simplified_positives', 'negatives_simplified_negatives', 'simplified_negatives', 'simplified_positives', 'hand_negatives_positives']
__all__ = ['get_simplified_hand', 'get_simplified']
hand_negatives_positives = [
    'a photo of a jump that is expected and uncreative and easy entry',
    'a photo of a jump that is not containing clear recognizable (difficult for jump preceded by steps/movements of the Short Program) steps/free skating movements immediately preceding element',
    'a photo of a jump that is not containing varied position in the air / delay in rotation',
    'a photo of a jump that is containing poor height or distance',
    'a photo of a jump that is containing poor extension on landing / creative exit',
    'a photo of a jump that is containing poor flow from entry to exit including jump combinations / sequences',
    'a photo of a jump that is effortful throughout',
    'a photo of a jump that is not matched to the musical structure',
    'a photo of a spin that is containing poor speed and acceleration during spin',
    'a photo of a spin that is not centering a spin quickly',
    'a photo of a spin that is containing unbalanced rotation in at least one position',
    'a photo of a spin that is at most the required number of revolutions',
    'a photo of a spin that is not containing good, strong position(s) (including height and air/landing position in flying spins)',
    'a photo of a spin that is uncreative or unoriginal',
    'a photo of a spin that is poor control in at least one phase',
    'a photo of a spin that is not matched to the musical structure',
    'a photo of a step sequence that is containing poor energy or execution',
    'a photo of a step sequence that is containing poor speed and acceleration during sequence',
    'a photo of a step sequence that is containing the use of poor executed various steps during the sequence',
    'a photo of a step sequence that is not containing deep clean edges',
    'a photo of a step sequence that is indicating poor control or commitment of the whole body or inaccurate steps',
    'a photo of a step sequence that is not creative and original',
    'a photo of a step sequence that is effortful',
    'a photo of a step sequence that does not enhance the musical structure',
    'a photo of a choreographic step sequence that is not containing good flow, energy or execution',
    'a photo of a choreographic step sequence that is containing poor speed and acceleration during sequence',
    'a photo of a choreographic step sequence that is or has containing poor clarity or precision',
    'a photo of a choreographic step sequence that is or has containing poor control or commitment of whole body',
    'a photo of a choreographic step sequence that is or has is uncreative or original',
    'a photo of a choreographic step sequence that is or has effortful',
    'a photo of a choreographic step sequence that is not reflecting concept or character of the program',
    'a photo of a choreographic step sequence that is not enhancing the musical structure'
]


simplified_positives =[
    'jump that is unexpected or creative or difficult entry',
    'jump with clear recognizable steps/free skating movements immediately preceding element',
    'jump with varied position in the air / delay in rotation',
    'jump with good height and distance',
    'jump with good extension on landing or creative exit',
    'jump with good flow from entry to exit including jump combinations / sequences',
    'jump that is effortless throughout',
    'jump matched to the musical structure',
    'spin with good speed or acceleration during spin',
    'centering a spin quickly',
    'spin with balanced rotations in all positions',
    'spin with clearly more than required number of revolutions',
    'spin good, strong position(s) and height and air/landing position in flying spins',
    'creative and original spin',
    'good control throughout all phases of spin',
    'spin matched to the musical structure',
    'step sequence with good energy and execution',
    'good speed or acceleration during step sequence',
    'well executed various steps during the step sequence',
    'step sequence with deep clean edges including entry and exit of all turns',
    'good control and commitment of the whole body maintaining accuracy of steps in step sequence',
    'creative and original step sequence',
    'effortless throughout step sequence',
    'step sequence enhances the musical structure',
    'choreographic step sequence with good flow, energy and execution',
    'good speed or acceleration during choreographic step sequence',
    'choreographic step sequence with good clarity and precision',
    'choreographic step sequence with good control and commitment of whole body',
    'creative and original choreographic step sequence',
    'effortless throughout choreographic step sequence',
    'choreographic step sequence reflecting concept or character of the program',
    'choreographic step sequence enhancing the musical structure'
]




simplified_negatives = [
    'combo contained only one jump',
    'downgraded jump',
    'no required preceding steps/movements prior to jump',
    'Under-rotated jump',
    'break between required steps/movements & jump/only 1 step/movement preceding jump',
    'jump lacking rotation including half loop in a combo',
    'fall during jump',
    'jump has poor speed, height, distance, air position',
    'landing on two feet in a jump',
    'touch down with both hands in a jump',
    'stepping out of landing in a jump',
    'touch down with one hand or free foot in a jump',
    '2 three turns in between in a jump combo',
    'loss of flow/direction/rhythm between jumps in combo or sequence',
    'severe wrong edge take off in jump',
    'jump with weak landing (bad pos./wrong edge/scratching etc)',
    'jump with unclear or wrong edge take off',
    'jump with poor take-off',
    'jump with long preparation',
    'flying spin where prescribed air position not attained',
    'spin with poor/awkward, unaesthetic position(s)',
    'fall during spin',
    'traveling during spin',
    'spin where there is a touch down with both hands',
    'slow or reduction of speed during spin',
    'spin where there is a touch down with free foot or one hand',
    'spin with change of foot poorly done',
    'spin with less than required revolutions',
    'incorrect take-off or landing in a flying spin',
    'flying spin with poor fly',
    'step sequence with jumps with more than half revolution included',
    'step sequence with poor quality of steps, turns, positions',
    'fall during step sequence',
    'stumble during step sequence',
    'less than half of the pattern doing steps/turns during step sequence',
    'step sequence does not correspond to the music',
    'fall during choreographic sequence',
    'stumble during choreographic sequence',
    'inability to clearly demonstrate the sequence during choreographic sequence',
    'choreographic sequence does not enhance the music',
    'loss of control while executing the choreographic sequence',
    'poor quality of movements during choreographic sequence'
]

negatives_simplified_negatives = [
    'combo contained two jumps',
    'successfully jump',
    'completed required preceding steps/movements prior to jump',
    'correctly rotated jump',
    'no break between required steps/movements and jump or only 1 step or movement preceding jump',
    'jump has full rotation including half loop in a combo',
    'no fall',
    'jump has good speed, height, distance, air position',
    'landing on one foot in a jump',
    'touch down at most one hand in a jump',
    'not stepping out of landing in a jump',
    'touch down with no hand and free foot in a jump',
    'at most one three turns in between in a jump combo',
    'no loss of flow/direction/rhythm between jumps in combo or sequence',
    'no or slight wrong edge take off in jump',
    'jump with strong landing',
    'jump with clear and right edge take off',
    'jump with good take-off',
    'jump with short preparation',
    'flying spin where prescribed air position is attained',
    'spin with good/graceful, aesthetic position(s)',
    'no fall during spin',
    'in place during spin',
    'spin where there is no touch down with both hands',
    'not slow and maintain speed during spin',
    'spin where there is no touch down with free foot or one hand',
    'spin with change of foot well done',
    'spin with at least required revolutions',
    'correct take-off and landing in a flying spin',
    'flying spin with good fly',
    'step sequence with jumps with at most half revolution included',
    'step sequence with good quality of steps, turns, positions',
    'no fall during step sequence',
    'no stumble during step sequence',
    'at least half of the pattern doing steps and turns during step sequence',
    'step sequence corresponds to the music',
    'no fall during choreographic sequence',
    'no stumble during choreographic sequence',
    'clearly demonstrate the sequence during choreographic sequence',
    'choreographic sequence enhances the music',
    'control while executing the choreographic sequence',
    'high quality of movements during choreographic sequence'
]

hand_negatives_simplified_positives = [
    'jump that is expected and uncreative and easy entry',
    'jump with unclear recognizable steps/free skating movements immediately preceding element',
    'jump without varied position in the air / delay in rotation',
    'jump with poor height or distance',
    'jump with poor extension on landing and creative exit',
    'jump with poor flow from entry to exit including jump combinations / sequences',
    'jump that is effortful throughout',
    'jump not matched to the musical structure',
    'spin with poor speed or acceleration during spin',
    'centering a spin slowly',
    'spin with unbalanced rotations in at least one position',
    'spin with at most the required number of revolutions',
    'spin with poor or weak position(s), height, or air/landing position in flying spins',
    'uncreative or unoriginal spin',
    'poor control at any phase of spin',
    'spin not matched to the musical structure',
    'step sequence with poor energy or execution',
    'poor speed and acceleration during step sequence',
    'poorly executed steps during the step sequence',
    'step sequence without deep clean edges including entry or exit of a turn',
    'poor control or commitment or inaccurate steps in step sequence',
    'uncreative or unoriginal step sequence',
    'effortful throughout step sequence',
    'step sequence does not enhance the musical structure',
    'choreographic step sequence with poor flow, energy or execution',
    'poor speed and acceleration during choreographic step sequence',
    'choreographic step sequence with poor clarity or precision',
    'choreographic step sequence with poor control or commitment of whole body',
    'uncreative or unoriginal choreographic step sequence',
    'effortful throughout choreographic step sequence',
    'choreographic step sequence not reflecting concept or character of the program',
    'choreographic step sequence not enhancing the musical structure'
]

def get_simplified(positive=False):
    if positive:
        return simplified_positives
    else:
        return simplified_negatives

def get_simplified_hand(positive_rubric=False, simplified=False):
    if positive_rubric:
        if simplified:
            return hand_negatives_simplified_positives
        else:
            return hand_negatives_positives
    else:
        if simplified:
            return negatives_simplified_negatives
        else:
            raise Exception("No hand negatives for non-simplified negatives")