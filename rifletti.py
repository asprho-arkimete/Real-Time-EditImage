import os
from PIL import Image

path = r"./pose/breast inv"
output_path = os.path.join(path, "s")

os.makedirs(output_path, exist_ok=True)

for f in os.listdir(path):
    if f.lower().endswith(".png"):
        input_file = os.path.join(path, f)
        output_file = os.path.join(output_path, f)

        with Image.open(input_file) as img:
            riflessa = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            riflessa.save(output_file)

        print(f"Riflesso: {f}")

print("Operazione completata!")