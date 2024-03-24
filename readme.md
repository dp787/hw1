# Homework 1 : Create an Agent to perform as a particular job.  I did an example of history teacher and movie advisor, you need to use this repository and watch the **lecture video** [here](https://youtu.be/T2h6mSycMqg).  What I want you to do is practice writing system prompts and see how the agent will obey your commands (or not).  You should change the model to use other models to experiment but you also need to look at how much they cost:

* List of GPT4 Models - [here](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
* List of GPT3.5 Models - [here](https://platform.openai.com/docs/models/gpt-3-5-turbo)

* Model Pricing - PAY ATTENTION IT COSTS A LOT FOR SOME MODELS.
[Pricing is here but scroll down and look at what GPT4 preview costs $60 for 1 million tokens](https://openai.com/pricing)

## Project Setup

1. Clone the repository.
2. CD into the project folder.
3. Create and activate the virtual environment (VE).
4. Install the required libraries.
5.  Signup for OpenAI Api Key.  Make .env file and put the key in as "open_ai_key=YOUR_KEY"

## Testing Commands

- Run all tests with `pytest`.
- To test a specific file, use `pytest tests/test_main.py`.
- For linting and coverage, `pytest --pylint --cov` commands can be used separately.

## Installed Libraries

1. [Pytest](https://docs.pytest.org/en/8.0.x/)
2. [Faker](https://faker.readthedocs.io/en/master/)
3. [Pytest Coverage](https://pytest-cov.readthedocs.io/en/latest/readme.html)
4. [Pytest Pylint](https://pylint.readthedocs.io/en/stable/development_guide/contributor_guide/tests/launching_test.html)
5. [Langchain](https://python.langchain.com/docs/get_started/quickstart)


## Adding a Library

1. Ensure you're in the correct VE; if unsure, run "deactivate".
2. Activate the VE.
3. Update the requirements file with `pip freeze > requirements.txt`.

