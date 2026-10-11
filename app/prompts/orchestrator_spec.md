Write a training-data specification for the plan below. Output ONLY JSON, with no explanation.

[Training plan]
{{plan}}

[Output format]
Replace every <...> with real content for this plan. Do not copy the descriptions.
{
  "language": "<language of the training data>",
  "system_prompt": "<a short system prompt shared by all student training examples>",
  "answer_format": "<format rules every answer must follow>",
  "avg_answer_tokens": <integer>,
  "categories": [
    {
      "name": "<category name>",
      "description": "<what this category covers>",
      "count": <integer>,
      "difficulty": {"easy": <0-1>, "medium": <0-1>, "hard": <0-1>},
      "example_question": "<an example question>"
    }
  ]
}

Rules:
- The sum of all category counts must equal {{target_samples}}.
- Use 3 to 12 categories that do not overlap.
- The difficulty ratios of each category must sum to 1.
- Write "name", "description", and "answer_format" in ENGLISH (the teachers read them in English).
- Write "system_prompt" and "example_question" in the training-data language ("language"), because they become part of the student's data.
