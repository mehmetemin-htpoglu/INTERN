import ollama

girdi = input("\nLütfen sorunuzu giriniz:\n")

stream = ollama.generate(
    model='deepseek-r1:14b',
    prompt= girdi,
    stream=True,
)

for parca in stream:
    print(parca['response'], end='', flush=True)