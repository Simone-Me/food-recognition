"""Write data/labels/foodseg103.json (id -> name). Committed: it is part of the problem definition.

    python -m scripts.make_labels

Source: the `id | ingridient` table of the dataset card of EduardoPacheco/FoodSeg103 (same order as the
official repo LARC-CMU-SMU/FoodSeg103-Benchmark-v1). VERIFY it against the card, then check visually
with scripts.show_samples: a "pizza" label on a salad means the table is shifted by one.
"""
from src.utils import ROOT, save_json

FOODSEG103 = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding",
    "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond",
    "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot",
    "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach",
    "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork",
    "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup",
    "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice",
    "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed",
    "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot",
    "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage",
    "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake",
    "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients",
]

if __name__ == "__main__":
    assert len(FOODSEG103) == 104, len(FOODSEG103)
    out = ROOT / "data" / "labels" / "foodseg103.json"
    save_json({str(i): n for i, n in enumerate(FOODSEG103)}, out)
    print(f"wrote {out} ({len(FOODSEG103)} entries). Spot check: 3={FOODSEG103[3]}, 60={FOODSEG103[60]}, "
          f"61={FOODSEG103[61]}")
