from neural_search import NeuralSearch


def search(query: str) -> None:
    search_module = NeuralSearch()
    search_module.predict(query)


if __name__ == "__main__":
    search("return code_series.progress_apply(lambda x: self.get_doc_embedding(x))")
