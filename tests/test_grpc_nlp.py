"""
Script de prueba del servidor gRPC NLP.
Ejecutar con el servidor corriendo en otra terminal.
"""

import sys
import os
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "api", "protos"))

import src.api.protos.nlp_pb2 as nlp_pb2
import src.api.protos.nlp_pb2_grpc as nlp_pb2_grpc


def test_servidor():
    """Envia una solicitud de prueba al servidor gRPC NLP."""
    channel = grpc.insecure_channel("localhost:50052")
    stub = nlp_pb2_grpc.NLPServiceStub(channel)

    request = nlp_pb2.RecommendationRequest(
        class_name="Tomato___Late_blight",
        confidence=0.954,
    )

    print("Enviando solicitud al servidor gRPC NLP...")
    response = stub.GetRecommendation(request)

    print(f"Success: {response.success}")
    print(f"Error: {response.error}")
    print(f"Recomendacion:\n{response.recommendation}")


if __name__ == "__main__":
    test_servidor()
