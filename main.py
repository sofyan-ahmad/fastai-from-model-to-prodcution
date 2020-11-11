from fastai.learner import load_learner
from fastai.vision.core import PILImage

model_inf = load_learner('./model_files/export.pkl')

img = PILImage.create('./test_data/001.jpeg')
what, _, probs = model_inf.predict(img)

print(f"001")

print(f"What is this?: {what}.")
print(f"Probability it's a {what}: {probs[1].item():.6f}")

print()

img = PILImage.create('./test_data/002.jpeg')
what, _, probs = model_inf.predict(img)

print(f"002")
print(f"What is this?: {what}.")
print(f"Probability it's a {what}: {probs[1].item():.6f}")
