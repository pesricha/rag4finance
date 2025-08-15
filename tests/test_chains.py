"""
This can also be used to print the models in ollama.
"""

import unittest
import requests


class TestOllamaServer(unittest.TestCase):
    def test_ollama_server_running(self):
        """
        Tests if the Ollama server is running by making a request to the /api/tags endpoint.
        Note that once ollama is installed it keeps on running in the background.
        """
        try:
            # Added a timeout of 5 seconds to fail faster if the server is unresponsive
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            response.raise_for_status()
            self.assertIn(
                "models", response.json(), "Invalid response format from Ollama server"
            )
        except requests.exceptions.ConnectionError:
            self.fail(
                "Could not connect to the Ollama server at http://127.0.0.1:11434. Please ensure it is running."
            )
        except requests.exceptions.Timeout:
            self.fail(
                "The request to the Ollama server timed out. The server might be running but unresponsive."
            )
        except requests.exceptions.RequestException as e:
            self.fail(
                f"An error occurred while communicating with the Ollama server: {e}"
            )


if __name__ == "__main__":
    # This block will not be executed when running tests with `python -m unittest`
    # It allows the file to be run directly to check for available models.
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models")
        if models:
            print("Available Ollama models:")
            for model in models:
                print(f"- {model['name']}")
        else:
            print("No models found on the Ollama server.")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to Ollama server: {e}")
