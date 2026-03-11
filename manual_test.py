from src.vision.classifier import classify_image

def test():
    print("Iniciando clasificacion...")
    res = classify_image('hojas-amarillas-plantas.jpg')
    print("RESULTADO:", res)

if __name__ == "__main__":
    test()
