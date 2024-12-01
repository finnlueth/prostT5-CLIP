import matplotlib.pyplot as plt

aspect_missing_counts = {}
results_path = 'C:/Users/ameli/OneDrive/Dokumente/results.txt'

with open(results_path, 'r') as file:
    current_aspect = None
    for line in file:
        line = line.strip()
        if line.startswith("Aspect:"):
            current_aspect = line.split(": ")[1]
        elif current_aspect and line.startswith("Missing terms ("):
            try:
                count = int(line.split("(")[1].split(")")[0])
                aspect_missing_counts[current_aspect] = count
            except (IndexError, ValueError):
                print(f"Error parsing line: {line}")

aspects = list(aspect_missing_counts.keys())
missing_counts = list(aspect_missing_counts.values())

colors = ['blue', 'red', 'green'] * (len(aspects) // 3 + 1)
colors = colors[:len(aspects)]

plt.figure(figsize=(10, 6))
plt.bar(aspects, missing_counts, color=colors)
plt.xlabel('Aspect')
plt.ylabel('Number of Missing GO-Terms')
plt.title('Missing GO-Terms by Aspect')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
