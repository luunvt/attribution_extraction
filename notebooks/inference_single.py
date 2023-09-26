from transformers import pipeline
from collections import defaultdict

model_checkpoint = "./distilbert-base-uncased_v3"
token_classifier = pipeline(
    "ner", model=model_checkpoint, aggregation_strategy="simple"
)

description = '''
Any Where - Any Time Workout: Sculpt your ideal body with solid cast iron kettlebells that last a lifetime - no welds, weak spots, or seams
Comfortable Workout: Neoprene coating prevents corrosion, reduces noise, protects flooring, and enhances appearance - Great for indoor and outdoor training
Secure Workout: The smooth, wide textured handle of kettlebell provides a comfortable and secure grip for high reps, making chalk no longer necessary
Personalized Workout: Color coded by weight for more excitement and energy - Choose your favorite colors for every session
Transform Your Body By Workout: Whether you're a beginner or an experienced athlete, our kettlebells are perfect for any fitness level'''
results = token_classifier(description)
print(results)
ans = defaultdict(set)
for res in results:
    group = res['entity_group']
    ans[group].add(res['word'].lower())

for key in ans.keys():
    line = f"{key}: {ans[key]}"
    print(line)