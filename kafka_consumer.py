from services import consume_predictions


def main():
    print("Listening for prediction messages...")
    for message in consume_predictions():
        print("Received message:", message)


if __name__ == "__main__":
    main()
