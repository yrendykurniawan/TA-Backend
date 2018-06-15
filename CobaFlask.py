from flask import Flask
import graphene
from graphene import ObjectType, String, Schema, Field
from flask_graphql import GraphQLView


class Query(ObjectType):
    hello = graphene.String(name=graphene.String(default_value="stranger"))

    def resolve_hello(self, info, name):
        return 'Hello ' + name


view_func = GraphQLView.as_view('graphql', schema=Schema(query=Query))

app = Flask(__name__)
app.add_url_rule('/', view_func=view_func)

if __name__ == '__main__':
    app.run()