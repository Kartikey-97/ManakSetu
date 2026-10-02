import os
os.environ["QDRANT_URL"] = "https://79b455c4-f081-4460-8344-317691614c5c.eu-west-2-0.aws.cloud.qdrant.io"
os.environ["QDRANT_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTBjYWYyZjktNzQxNC00MzMzLThhNTktMTFiYzJiYzgzZTZlIn0.tnoZDtkavTaQRUd9JbBWtAp8s9-LMQ5dYAxOEumFjXc"

from qdrant_client import QdrantClient
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
print(client.get_collections())
