Evaluate a student language model's answer before training and after training. The reference answer is the accepted model answer.
The order is random; you are not told which answer is from after training.

[Question]
{{question}}

[Reference answer]
{{reference}}

[Answer X]
{{answer_x}}

[Answer Y]
{{answer_y}}

Score each answer from 1 to 10 for accuracy, format, and usefulness compared with the reference.
Output ONLY JSON. Replace every <...> with a real value.
{"X": <integer 1-10>, "Y": <integer 1-10>, "reason": "<one sentence>"}
