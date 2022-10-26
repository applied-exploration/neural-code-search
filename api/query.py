from flask_restful import Resource, reqparse
import pandas as pd
import ast
from neural_search import NeuralSearch


class Query(Resource):
    def __init__(self):
        self.search_module = NeuralSearch()

        print("")

    def get(self):

        parser = reqparse.RequestParser()  # initialize

        parser.add_argument("code", required=True, location="args")

        args = parser.parse_args()  # parse arguments to dictionary
        print(args)
        code_query = args["code"]
        k_best_indicies, original_snippets = self.search_module.predict(code_query)
        print(original_snippets)
        return {
            "data": {
                "k_best_indicies": k_best_indicies,
                "original_snippets": original_snippets[
                    [
                        "question_id",
                        "question_date",
                        "question_score",
                        "view_count",
                        "question_body",
                        "title",
                        "tags",
                        "answer_count",
                        "favorite_count",
                        "answer_date",
                        "answer_id",
                        "answer_score",
                        "answer_body",
                    ]
                ].to_json(),
            }
        }, 200  # return data and 200 OK code
