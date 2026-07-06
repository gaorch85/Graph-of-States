import json
import re
text = """{
  "candidates": [
    {
      "id": "h001",
      "label": "Inflammatory skin condition (e.g., lichen planus)",
      "confidence": 0.6,
      "why": "Node-2 morphology (violaceous annular patches without scale) and node-3 severe pruritus support; node-4 demonstrates resistance to mild steroid, fitting a more stubborn inflammatory process."
    },
    {
      "id": "h002",
      "label": "Cutaneous malignancy (e.g., lymphoma)",
      "confidence": 0.25,
      "why": "Node-3 severe nocturnal pruritus is a red flag; node-2 morphology can occur in lymphoma; node-4 steroid resistance is consistent, but localized nature reduces likelihood."
    },
    {
      "id": "h003",
      "label": "Superficial fungal infection",
      "confidence": 0.1,
      "why": "Despite annular shape, node-2 (no scale, Wood lamp negative) and node-4 (ketoconazole failure) strongly refute; node-3 pruritus is non-specific."
    },
    {
      "id": "h004",
      "label": "Allergic/Irritant contact dermatitis",
      "confidence": 0."""


match = re.search(r"""\"[a-z|A-Z]*\": \[""", text[idx:])


print(match.group(0))