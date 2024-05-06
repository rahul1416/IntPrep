import requests
import re
import json
def generate_response(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        if "response" in response_data:
            return response_data["response"]
        else:
            return "No response field found in the response content."
    else:
        return f"Error: {response.text}"

model = "interview"
prompt_1 = "{question:Can you explain your experience in building and fine-tuning chatbots and NLP systems, specifically within the context of the travel domain?}"
prompt_2 = "{answer: Experience in Building Chatbots and NLP Systems: I have over 7 years of experience in software engineering, with a focus on NLP and chatbot development for the past 3 years. Specifically within the travel domain, I've built chatbots that assist with itinerary planning, booking, and customer support. These systems leverage NLP techniques to understand user queries and provide relevant responses.}"
prompt = prompt_1 + prompt_2
response_content = generate_response(model, prompt)
print("Response Content:", response_content)

integers = re.findall(r'\d+', response_content)

print(integers)
print(integers[0])
integer_part = ''.join(integers)

print("Integer Part:", integer_part)
