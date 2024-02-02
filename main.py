from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")


from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Function to answer questions
def answer_question(question, model=model, tokenizer=tokenizer):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def solve_homework(problem_description, model=model, tokenizer=tokenizer):
    prompt = f"Help me solve this homework problem: {problem_description}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    solution_steps = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return solution_steps

# Function to explain concepts
def explain_concept(concept, model=model, tokenizer=tokenizer):
    prompt = f"Explain the concept of {concept}:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation

def create_essay_outline(topic, model=model, tokenizer=tokenizer):
    prompt = f"Create an essay outline for the topic: {topic}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1)
    essay_outline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return essay_outline

def tutor_subject(question, subject, model=model, tokenizer=tokenizer):
    prompt = f"As a {subject} tutor, answer this: {question}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    subject_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return subject_answer

# Function for language practice (e.g., translation)
def translate_text(text, target_language, model=model, tokenizer=tokenizer):
    prompt = f"Translate this to {target_language}: {text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def main():
    print("Welcome to Phi-Learn Assistant. Type 'exit' to leave the program.")
    while True:
        user_input = input("How can I assist you with your learning today? ")
        if user_input.lower() == 'exit':
            break
        elif "explain" in user_input.lower():
            concept = user_input.replace("explain", "").strip()
            print(explain_concept(concept))
        elif "translate" in user_input.lower():
            # This is a simplification, you would need to parse the language and text
            text = user_input.replace("translate", "").strip()
            print(translate_text(text, "Spanish"))  # Assuming you want to translate to Spanish
        else:
            print(answer_question(user_input))

if __name__ == "__main__":
    main()