cache = {}

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    # check cache
    if query in cache:
        print("\nAnswer from cache:")
        print(cache[query])
        continue

    # fake answer for now
    answer = f"Generated answer for: {query}"

    # save to cache
    cache[query] = answer

    print("\nNew answer:")
    print(answer)