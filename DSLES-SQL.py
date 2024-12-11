import pandas as pd
import time
import os
import sys
import dashscope
from dashscope.api_entities.dashscope_response import Role
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
device="cuda"

# Here to load fine tuned model & tokenizer, please replace "model_path" as your own.
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("model_path")

def call_with_messages(prompt):
    messages=[]
    for index, text in enumerate(prompt):
        if index % 2 == 0:
            messages.append({'role': Role.USER,
                             'content': text})
        else:
            messages.append({'role': Role.ASSISTANT,
                             'content': text})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    # Generate the response
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id  # Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if sys.argv[1] == "--dataset" and sys.argv[3] == "--output":
    DATASET_SCHEMA = sys.argv[2]+"tables.json"
    DATASET = sys.argv[2]+"dev.json"
    OUTPUT_FILE = sys.argv[4]
else:
    raise Exception("Please use this format python DSLES-SQL.py --dataset ./data/spider/ --output predicted_sql.txt")




def load_data(DATASET):
    return pd.read_json(DATASET)


def std_generation(prompt):
  response = call_with_messages(prompt)
  return response.strip()


def index_of_max_value(lst):
    for index, value in enumerate(lst):
        if value == max(lst):
            return index


from sentence_transformers import SentenceTransformer, util

# Download model
Sentence_model = SentenceTransformer('./paraphrase-MiniLM-L6-v2')
examples = pd.read_json('./data/train_mask.json')
original_examples = pd.read_json('./data/spider/train_spider.json')

def example_select(mask_question,n=20):
    sentences=[]
    sentences.append(mask_question)
    selected_examples = []
    for index, row in examples.iterrows():
        example_mask_question = row['mask_question']
        sentences.append(example_mask_question)

    # Get embeddings of sentences
    embeddings = Sentence_model.encode(sentences)

    similarities_list = []
    for embedding in embeddings:
        similarities_list.append(util.cos_sim(embeddings[0],embedding).item())   #计算相似性

    similarities_list[0] = 0
    example_selected = []
    for i in range(n):
        index = index_of_max_value(similarities_list)
        example_selected.append({"question":original_examples.iloc[index-1, 4],"query":original_examples.iloc[index-1, 1],"db_id":original_examples.iloc[index-1, 0]})
        similarities_list[index] = 0
    print(example_selected)
    return example_selected

import re
def convert_number(s):
    for num in re.findall(r'\d+', s):
        try:
            number = int(num)
            return number
        except ValueError:
            return 1
    return 1

def ES(mask_question,question,database_schema,n):
    examples = example_select(mask_question)  # What are the names of the teachers who are aged either 32 or 33?
    prompt = []
    scores = []
    ES_examples = []

    for index, exam in enumerate(examples):
        txt = "The database ID to be queried is '" + exam['db_id'] + "'\n" + "Q" + str(index) + ":" + exam[
            'question'] + "\n" + "SQL" + str(index) + ":" + exam['query'] + "\n\n"

        txt = txt + "The database to be queried is:\n" + database_schema +"\n##The more inspiration the above example provides for the question '"+ mask_question +"', the higher the score will be.\n##The full score is 100. Please rate the example."
        prompt.append(txt)
        result = call_with_messages(prompt)
        prompt.append(result)
        print("result："+result)
        score = int(convert_number(result))
        scores.append(score)

    for i in range(n):
        index = index_of_max_value(scores)
        ES_examples.append(examples[index])
        scores[index] = 0
    return ES_examples



def SQL_GENERATION(ES_examples,question,dabatas_schema):
    prompt=['''

    Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.
    The database ID to be queried is \"college_2\".
    Q: "Find the buildings which have rooms with capacity more than 50."
    A: Let’s think step by step.''', '''
    In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
    "the buildings which have rooms" so we need column = [classroom.capacity]
    "rooms with capacity" so we need column = [classroom.building]
    Based on the columns and tables, we need these Foreign_keys = [].
    Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
    Schema_links: [classroom.building,classroom.capacity,50]''','''

Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.
The database ID to be queried is "student_assessment".
Q: "List the id of students who never attends courses?"
A: Let’s think step by step.''','''
In the question "List the id of students who never attends courses?", we are asked:
"id of students" so we need column = [Students.student_id]
"never attends courses" so we need column = [Student_Course_Attendance.student_id]
Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]''']

    txt = ''
    for index, exam in enumerate(ES_examples):
        txt = txt + "The database ID to be queried is '" + exam['db_id'] + "'\n" + "Q" + str(index) + ":" + exam[
            'question'] + "\n" + "SQL" + str(index) + ":" + exam['query'] + "\n\n"
    txt = txt + '''

    Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.
    The database to be queried is：
    ''' + dabatas_schema + '''
    A: Let’s think step by step.'''

    prompt.append(txt)
    schema_link = None
    while schema_link is None:
        try:
            schema_link = call_with_messages(prompt)
        except:
            print("666")
            time.sleep(3)
            pass
    try:
        schema_link = schema_link.split("Schema_links: ")[1].split("\n")[0]
    except:
        schema_link = ""

    prompt = []
    for index, exam in enumerate(ES_examples):
        txt = "##I have now provided the database schema and the query question.\n##Please generate the corresponding SQL statement.\n" \
              + "The database ID to be queried is '" + exam['db_id'] + "'\n" + "Q" + str(index) + ":" + exam[
            'question'] + "\n"
        prompt.append(txt)
        txt = exam['query'] + "\n"
        prompt.append(txt)

    txt =  "##I have now provided the database schema and the query question.\n##Please generate the corresponding SQL statement.\n" + dabatas_schema + schema_link
    prompt.append(txt)
    SQL = None
    while SQL is None:
        try:
            SQL = call_with_messages(prompt)
        except:
            time.sleep(3)
            print("SQL slicing error")
            SQL = "SELECT"

    print(SQL)
    return SQL

def debug_test():
    code_dev = load_data('./data/spider_code_dev_fintune.json')
    SQL_list = []
    CODEX = []
    with open('predicted_sql.txt',
              'r') as file:
        lines = file.readlines()
        for line in lines:
            SQL_list.append(line)
    for index, row in code_dev.iterrows():
        instruction = """#### I will give you the Database schema、 question and a preliminary SQL QUERY as input, please help me fix the given SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQL QUERY as is.Please remember that if you are not sure, do not modify the original SQL QUERY.
        #### Use the following instructions for fixing the SQL QUERY:
        1) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
        2) Use DESC and DISTINCT when needed.

        """

        answer = 'OK, I get it. I will help you fix the SQL QUERY according to your request. I will think about my reasons before fixing it, and if I am not 99% sure I can fix it, I will output your original SQL QUERY. My final output is just the corresponding SQL, without any additional output.'
        prompt = []
        prompt.append(instruction)
        prompt.append(answer)
        fields = row['input'] + '\n\nSQL QUERY:' + SQL_list[index]
        prompt.append(fields)
        print(fields)
        SQL = std_generation(prompt).replace('\n', '')
        print("fixed："+SQL)
        CODEX.append([row['input'], SQL])
    df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL'])
    results = df['PREDICTED SQL'].tolist()
    with open('fixed_predicted_sql.txt', 'w') as f:
        for line in results:
            f.write(f"{line}\n")

if __name__ == '__main__':
    code_dev = load_data('./data/spider_code_dev_fintune.json')
    mask_questions = load_data('./data/dev_mask.json')
    val_df = load_data(DATASET)
    print(f"Number of data samples {val_df.shape[0]}")
    CODEX = []
    for index, row in val_df.iterrows():


        print(f"index is {index}")
        print(row['query'])
        print(row['question'])
        ES_examples = ES(mask_questions['mask_question'][index],row['question'],code_dev.iloc[index, 1],n=10) #n=k
        SQL = SQL_GENERATION(ES_examples,row['question'],code_dev.iloc[index, 1])

        CODEX.append([row['question'], SQL, row['query'], row['db_id']])
        #break
    df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
    results = df['PREDICTED SQL'].tolist()
    with open(OUTPUT_FILE, 'w') as f:
        for line in results:
            f.write(f"{line}\n")

    #self-correction
    # debug_test()  




