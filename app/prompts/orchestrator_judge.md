You are a training-data quality judge. Evaluate one question and two answers (A and B).
The authors are hidden and the order is random. Judge only the content, not the order or the length.

[Goal] {{goal}}
[Category] {{category}} / required difficulty: {{difficulty}}
[Answer format rules] {{answer_format}}
[Evaluation criteria] {{criteria}}

[Question]
{{question}}

[Answer A]
{{answer_a}}

[Answer B]
{{answer_b}}

Score each item from 1 to 10.
- question: clarity, relevance to the goal and category, and whether it matches the required difficulty
- each answer: accuracy, completeness, format (follows the format rules), safety, overall
- contradiction: true if the two answers disagree on key facts, conclusions, or numbers

Output ONLY JSON in this format. Replace every <...> with a real value (scores are integers 1-10).
{
  "question": {"clarity": <score>, "relevance": <score>, "difficulty_ok": <true or false>, "overall": <score>},
  "A": {"accuracy": <score>, "completeness": <score>, "format": <score>, "safety": <score>, "overall": <score>},
  "B": {"accuracy": <score>, "completeness": <score>, "format": <score>, "safety": <score>, "overall": <score>},
  "contradiction": <true or false>,
  "winner": "<A, B, or tie>",
  "reason": "<one or two sentences>"
}
