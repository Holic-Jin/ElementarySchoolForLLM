You are Teacher 1. Create training items (a question and a model answer) to teach a student language model.

[Goal] {{goal}}
[Language] {{language}}
[Category] {{category}} — {{category_description}}
[Answer format] {{answer_format}}
[Example question] {{example_question}}

Create exactly {{n}} items with this difficulty mix: {{difficulty_plan}}
- Do not make only easy items. Follow the difficulty mix exactly.
- Every question must be different and sound like something a real user would ask.
- Do not repeat or paraphrase these existing questions: {{avoid}}
- Answers must be correct, complete, and follow the answer format.
- Write questions and answers in {{language}}.

Output ONLY this JSON, with no explanation:
{"items": [{"difficulty": "easy|medium|hard", "question": "...", "answer": "..."}]}
