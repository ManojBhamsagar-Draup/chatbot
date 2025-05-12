# chatbot
## Docker Setup and Running

This method packages the application and its dependencies into a container.

1.  **Ensure Prerequisites:** Make sure Docker is installed and running.
2.  **Build the Docker Image:**
    *   Open your terminal in the project root directory (`rag_chatbot/`).
    *   Run the build command:
        ```bash
        docker build -t rag-chatbot-gemini .
        ```
        (Replace `rag-chatbot-gemini` with your desired image name).

3.  **Run the Docker Container:**
    *   Run the container, passing the API key as an environment variable:
        ```bash
        docker run -p 8501:8501 \
               -e GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_GEMINI_API_KEY" \
               --name my-rag-app \
               rag-chatbot-gemini
        ```
    *   `-p 8501:8501`: Maps the container's port 8501 to your host machine's port 8501.
    *   `-e GOOGLE_API_KEY=...`: **Securely passes the API key to the container environment.** Replace the placeholder with your actual key.
    *   `--name my-rag-app`: (Optional) Assigns a name to the running container.
    *   `rag-chatbot-gemini`: The name of the image you built.

4.  Open your web browser and navigate to `http://localhost:8501`.
