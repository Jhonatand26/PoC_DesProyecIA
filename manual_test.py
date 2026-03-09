from src.vision.classifier import classify_image
from PIL import Image

def test():
    print("Creando img mock...")
    img = Image.new('RGB', (224, 224), color='green')
    img.save('test.jpg')
    print("Iniciando clasificacion...")
    res = classify_image('test.jpg')
    print("RESULTADO:", res)

if __name__ == "__main__":
    test()
