import pandas as pd
import re
from ollama import generate_response
def generate_final_scores(df, emotion_scores):
    # df['Emotions'] = df['Emotions'].apply(lambda x: x.strip(',').split(','))


    def sum_emotion_scores(emotion_list):
        return sum(emotion_scores[emotion] for emotion in emotion_list) / len(emotion_list)


    df['Total_Score'] = df['Emotions'].apply(sum_emotion_scores)

    score = []
    remark = []
    for i in range(len(df)):
        model = "interview"
        prompt_1 = df['Question'][i]
        prompt_2 = df['Transcribed_Text'][i]
        prompt = "question:" + prompt_1 + "answer:" + prompt_2
        response_content = generate_response(model, prompt)
        print("Response Content:", response_content)
        integers = re.findall(r'\d+', response_content)
        remark.append(response_content)
        print(integers)
        print(integers[0])
        score.append(integers[0])


    df['score'] = score
    df['remark'] = remark


    df['Final_score'] = ((5 * df['Total_Score'] + df['score'].astype(int)) / 15)*100
    final = df['Final_score'].sum()/len(df)
    return final


df = pd.read_csv('question/data.csv')

# Define emotion scores dictionary
emotion_scores = {
    'Angry': -2,
    'Disgust': -1,
    'Fear': 2,
    'Happy': 2,
    'Neutral': 1,
    'Sad': -1,
    'Surprise': 0.5,
    'No_face_detected': 0
}

# Function to generate final scores and remarks
# df = generate_final_scores(df, emotion_scores)

# print(df)
