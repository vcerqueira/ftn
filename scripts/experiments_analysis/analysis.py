from src.results.reading_files import ReadFiles

results = ReadFiles.read_all_files(horizon=None)

results.rank(axis=1).mean()

print(results.rank(axis=1).mean().sort_values())
print(results.shape)
