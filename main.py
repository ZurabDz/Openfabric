import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
from random import choice

# https://github.com/UKPLab/sentence-transformers/tree/master/examples/unsupervised_learning/SimCSE
# I think by finetuning `sciq` dataset on given model with SimCSE approach would yeld better results
# For now it can be evaluated on validation set 
model = SentenceTransformer('all-MiniLM-L6-v2')


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


def precompute_embeddings():
    dataset = load_dataset("sciq")

    possible_questions = [value['question'] for value in dataset['test']]

    question_answer_mapper = {
        value['question']: value['correct_answer'] for value in dataset['train']}
    questions = list(question_answer_mapper.keys())
    corpus_embeddings = model.encode(questions, convert_to_tensor=True)

    return corpus_embeddings, questions, question_answer_mapper, possible_questions


corpus_embeddings, questions, question_answer_mapper, possible_questions = precompute_embeddings()


def find_match(query):
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    # For know choose top 1 result
    top_results = torch.topk(cos_scores, k=1)
    scores, indicies = top_results[0], top_results[1]

    return scores[0].detach().item(), indicies[0].detach().item()


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        score, indicie = find_match(text)
        if score < 0.5:
            answer = f'''test: {question_answer_mapper[questions[indicie]]} Sorry I am dumb, please try to ask me question about science, like:''' + \
            f''' {choice(possible_questions)}'''
        elif score < 0.6:
            answer = f'I am not sure the answer might be:  {question_answer_mapper[questions[indicie]]}'
        else:
            answer = f'The answer is: {question_answer_mapper[questions[indicie]]}'

        output.append(answer)

    return SimpleText(dict(text=output))
