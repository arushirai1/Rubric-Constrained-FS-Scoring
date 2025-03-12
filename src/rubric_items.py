from elements_preprocessing.hand_negatives import get_simplified, get_simplified_hand

class RubricItems:
    def __init__(self, simplified=False, text_prompt=False, is_rg=False):
        self.simplified = simplified
        self.text_prompt=text_prompt
        self.is_rg = is_rg
        if is_rg:
            from RG.deduction_criteria import get_all_positive_criteria_in_one, get_all_negative_criteria_in_one
        else:
            from elements_preprocessing.goe_criteria import get_all_positive_criteria_in_one, get_all_negative_criteria_in_one


        self.get_all_positive_criteria_in_one = get_all_positive_criteria_in_one
        self.get_all_negative_criteria_in_one = get_all_negative_criteria_in_one
    def get_positives(self):
        if self.simplified:
            if self.is_rg:
                positive_rubric_items, points = self.get_all_positive_criteria_in_one()
                return self.apply_text_prompt(positive_rubric_items), points # should be placeholder and 0
            else:
                return self.apply_text_prompt(get_simplified(positive=True))
        else:
            return self.apply_text_prompt(get_all_positive_criteria_in_one())
    def get_negatives(self):
        if self.simplified:
            if self.is_rg:
                negative_rubric_items, deductions = self.get_all_negative_criteria_in_one()
                return self.apply_text_prompt(negative_rubric_items), deductions
            else:
                return self.apply_text_prompt(get_simplified(positive=False))
        else:
            return self.apply_text_prompt(self.get_all_negative_criteria_in_one()[0])
    def get_hand_negatives(self, positive_rubric):
        return self.apply_text_prompt(get_simplified_hand(positive_rubric=positive_rubric, simplified=self.simplified))
    def apply_text_prompt(self, text_list):
        if not self.text_prompt:
            return text_list
        else:
            text_prompt = f"a photo of a %s"
            return [text_prompt.format(text) if text_prompt not in text else text for text in text_list]