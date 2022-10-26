from flask import Flask
from flask_restful import Api

from api.query import Query

# from endpoints.locations import Locations

app = Flask(__name__)
api = Api(app)


api.add_resource(Query, "/query")


if __name__ == "__main__":
    app.run()
