
import requests

BASE_URL = "http://127.0.0.1:8004"

def main():
    print("\n" + "=" * 55)
    print("       AVIATION SLM V3 - RAG CHATBOT")
    print("=" * 55)
    print("Model: aviation-slm-v3-rag:latest")
    print("RAG: Enabled")
    print("Type 'exit' to close the chatbot.")
    print("-" * 55)

    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
        print("Connection: Successful")
    except requests.RequestException:
        print("ERROR: RAG service is not running.")
        print("Please start rag_service.py first.")
        input("\nPress Enter to close...")
        return

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit", "bye"]:
                print("\nChatbot closed. Goodbye!")
                break

            if not question:
                continue

            response = requests.post(
                f"{BASE_URL}/chat",
                json={"question": question},
                timeout=120
            )

            response.raise_for_status()
            result = response.json()

            print("\nAviation SLM:", result.get("answer", "No answer received."))

        except requests.Timeout:
            print("\nThe request timed out. Please try again.")
        except requests.RequestException as error:
            print(f"\nConnection error: {error}")

if __name__ == "__main__":
    main()
