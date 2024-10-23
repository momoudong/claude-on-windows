# claude-on-windows

A Windows-compatible version of [Anthropic's Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), modified to run natively on Windows without requiring Docker containers.

## ⚠️ Caution

Computer use is a beta feature. Please be aware that computer use poses unique risks that are distinct from standard API features or chat interfaces. These risks are heightened when using computer use to interact with the internet. To minimize risks, consider taking precautions such as:

- Use a dedicated virtual machine with minimal privileges to prevent direct system attacks or accidents
- Avoid giving the model access to sensitive data, such as account login information, to prevent information theft
- Limit internet access to an allowlist of domains to reduce exposure to malicious content
- Ask a human to confirm decisions that may result in meaningful real-world consequences as well as any tasks requiring affirmative consent, such as accepting cookies, executing financial transactions, or agreeing to terms of service

In some circumstances, Claude will follow commands found in content even if it conflicts with the user's instructions. For example, instructions on webpages or contained in images may override user instructions or cause Claude to make mistakes. We suggest taking precautions to isolate Claude from sensitive data and actions to avoid risks related to prompt injection.

**EXPERIMENTAL SOFTWARE - USE AT YOUR OWN RISK**. This software is provided "as is", without warranty of any kind. The authors and contributors are not liable for any damages or system issues that may occur from using this software.

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `.\venv\Scripts\activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Configure your `.env` file with your Anthropic API key and screen resolution
6. Run the application: `python -m streamlit run run.py`

## Usage

1. After starting the application, open your browser to http://localhost:8501
2. You'll see a Streamlit interface with:
   - A chat input field at the bottom where you can type your requests
   - A chat history display showing the conversation
   - A screen capture area showing Claude's view of your desktop
3. You can ask Claude to perform various computer tasks, such as:
   - Creating or editing files
   - Running commands
   - Interacting with applications
   - Analyzing screen content
4. Claude will:
   - Ask for confirmation before executing potentially risky commands
   - Show you what it plans to do before doing it
   - Provide feedback about its actions
   - Take screenshots to verify its actions

**Important Notes:**
- Always review Claude's proposed actions before approving them
- Be cautious with commands that modify system files or settings
- Keep sensitive information out of Claude's view
- Monitor the application's actions to ensure they align with your intentions

## Screen Resolution

Environment variables `WIDTH` and `HEIGHT` in the `.env` file can be used to set the screen size. We do not recommend sending screenshots in resolutions above XGA/WXGA to avoid issues related to image resizing.

## Attribution

This project is based on [Anthropic's Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) and has been modified to run natively on Windows systems without requiring Docker containers. All credit for the original implementation goes to Anthropic.
