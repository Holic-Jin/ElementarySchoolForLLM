Finalize the training plan based on the conversation so far. Output ONLY the JSON below, with no explanation.
Replace every <...> with real content for this plan. Do not copy the descriptions.

IMPORTANT: Write every value in ENGLISH, even if the conversation was in another language (translate it).
The only exceptions are "language" (the name of the training-data language, written in English, e.g. "Korean")
and "user_summary_markdown" (written in the language the user used in the conversation).

{
  "title": "<short project title in English>",
  "goal": "<training goal in one or two English sentences>",
  "goal_type": "<knowledge or adaptation>",
  "domain": "<domain and scope, in English>",
  "language": "<language of the training data, e.g. Korean>",
  "answer_style": "<required answer format, tone, and length, in English>",
  "target_samples": <integer>,
  "student_model_comment": "<your opinion on the student model, in English>",
  "evaluation_criteria": ["<criterion 1 in English>", "<criterion 2 in English>"],
  "plan_markdown": "<the full plan in English Markdown: goal, scope, data composition, evaluation criteria, cautions>",
  "user_summary_markdown": "<a short summary of the plan in the user's conversation language>"
}

target_samples is an integer. If the user did not decide, recommend a value between 100 and 2000.
