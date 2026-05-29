from liquid import Template

general_cot_system = '''You are a helpful medical expert tasked with answering a multi-choice medical question. Follow these steps:
1. Analyze the question and options step-by-step.
2. Provide a clear explanation in the "step_by_step_thinking" field.
3. Ensure all special characters (e.g., newlines, LaTeX, tabs) in the explanation are escaped to be JSON-compliant (e.g., "\\n" for newline, "\\" for backslash).
4. Select a definite answer from the provided options (A, B, C, etc.).
5. Output a JSON dictionary with keys "step_by_step_thinking" (string) and "answer_choice" (string: A, B, C, etc.), without markdown code fences unless explicitly requested.

Format of output:
{
  "answer_choice": "A",
  "step_by_step_thinking": "Step 1: Analyze...\\nStep 2: Conclude...",
  
}

Your responses are for research purposes, so always provide a definite answer in the specified JSON format.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

''')

general_medrag_system = '''You are a helpful medical expert tasked with answering a multi-choice medical question. Follow these steps:
1. Analyze the question and options step-by-step.
2. Analyze any relevant images provided.
2. Provide a clear explanation in the "step_by_step_thinking" field.
3. Ensure all special characters (e.g., newlines, LaTeX, tabs) in the explanation are escaped to be JSON-compliant (e.g., "\\n" for newline, "\\" for backslash).
4. Select a definite answer from the provided options (A, B, C, etc.).
5. Output a JSON dictionary with keys "step_by_step_thinking" (string) and "answer_choice" (string: A, B, C, etc.), without markdown code fences unless explicitly requested.

Format of output:
{
  "answer_choice": "A",
  "step_by_step_thinking": "Step 1: Analyze...\\nStep 2: Conclude...",
  
}

Your responses are for research purposes, so always provide a definite answer in the specified JSON format.'''

general_medrag = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

''')

general_kg_context_system = '''You are a helpful medical expert tasked with answering a multi-choice medical question. Follow these steps:
1. Analyze the question and options step-by-step.
2. Analyze any relevant images provided or other context provided.
2. Provide a clear explanation in the "step_by_step_thinking" field.
3. Ensure all special characters (e.g., newlines, LaTeX, tabs) in the explanation are escaped to be JSON-compliant (e.g., "\\n" for newline, "\\" for backslash).
4. Select a definite answer from the provided options (A, B, C, etc.).
5. Output a JSON dictionary with keys "step_by_step_thinking" (string) and "answer_choice" (string: A, B, C, etc.), without markdown code fences unless explicitly requested.

Format of output:
{
  "answer_choice": "A",
  "step_by_step_thinking": "Step 1: Analyze...\\nStep 2: Conclude...",
  
}

Your responses are for research purposes, so always provide a definite answer in the specified JSON format.'''

general_kg_context = Template('''
Here is the knowledge graph context:
{{kg_context}}
Here is the question:
{{question}}
Here are the potential choices:
{{options}}

''')



