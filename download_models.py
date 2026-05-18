

import gdown
import os

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "best.pt": "1DMVto-mT0pC_6dmc7lJ3N6eCprTnDV_d",
    "efficientnet_best.keras": "1mwCENPnNGsokM3AeBNQ4DOacf8nqCFln",
    "mobilenet_fixed.h5": "1lFhrkJtfHdBtPmEC_WRZjYuCQNAOv8DF",
    "resnet50_best.keras": "1PL8hAaQELw9aY9mS1oioOgMQpN0l_9Od",
    "class_indices.json": "1Vejr0lSIAmhfuwMgtYaivvddTvu4Qk63",
    "efficientnet_weights.weights.h5": "1NwXJtB8ueTq-53Xy8g1FR9Ryo2X32sIb",
    "mobilenet_weights.weights.h5": "17346loi6ZccWl_yXydvv9R4r4HlyUs2C",
    "resnet50_weights.weights.h5": "1qU4nn3ihi05szB2Yo_mqZ6KsnU7os3Hf",
}


def download_models():

    for filename, file_id in MODELS.items():

        output = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(output):

            print(f"Downloading {filename}...")

            gdown.download(
                id=file_id,
                output=output,
                quiet=False
            )

        else:
            print(f"{filename} already exists.")

    print("All models downloaded successfully!")