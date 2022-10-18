from neural_search import NeuralSearch


def search(query: str) -> None:
    search_module = NeuralSearch()
    search_module.predict(query)


if __name__ == "__main__":
    search('def hello_s(s: str):\n    print(f"Hello s")')
