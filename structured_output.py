# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %% [markdown]
# # Structured Output Generation Workshop
#
# In this workshop, we'll explore different techniques for generating structured output using Large Language Models (LLMs).
# We'll cover:
# 1. Basic text to JSON conversion
# 2. Working with different LLM providers
# 3. Using Pydantic for type-safe parsing
# 4. Understanding token probabilities
# 5. Custom logits processing
# 6. Structured generation with Outlines

# %% [markdown]
# ## 1. Setting up our data sources
# First, let's create a helper function to fetch and cache our data.

# %%
import bs4
import requests


def get_text(url: str, cache_file: str) -> str:
    try:
        with open(cache_file) as file:
            text = file.read()
    except FileNotFoundError:
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text)
        with open(cache_file, "w") as file:
            file.write(soup.get_text())
    return text


planets_txt = get_text(
    "https://nssdc.gsfc.nasa.gov/planetary/factsheet/",
    "data/planets.txt",
)
satellites_txt = get_text(
    "https://ssd.jpl.nasa.gov/sats/phys_par/",
    "data/satellites.txt",
)

# %% [markdown]
# Let's examine our raw data:

# %%
print("=== Planets data ===")
print(planets_txt)

# %%
print("=== Satellites data ===")
print(satellites_txt)

# %% [markdown]
# ## 2. Basic Text to JSON Conversion
# Let's try converting our unstructured text data into JSON using an LLM.
# First, we'll set up our LLM client:

# %%
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# We'll start with Groq's API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

# %% [markdown]
# ### Exercise 1
# Try to write a prompt that would convert the planets data into JSON format.
# What challenges do you expect to face?

# %% [markdown]
# ### Solution

# %%
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": planets_txt},
    ],
)

response_content = response.choices[0].message.content
print(response_content)

# %% [markdown]
# Let's validate if we got valid JSON:

# %%
import json


def is_json(maybe_json: str):
    try:
        json.loads(maybe_json)
    except json.JSONDecodeError:
        return False
    return True


print(f"Is valid JSON? {is_json(response_content)}")

# %% [markdown]
# ### Exercise 2
# How reliable is JSON conversion with different prompts? Let's experiment!
#
# Try writing different prompts and test their reliability. Here's a function to help you evaluate your prompts:


# %%
def test_prompt_reliability(prompt: str, n_trials: int = 10) -> float:
    """
    Test how reliable a prompt is at generating valid JSON.

    Args:
        prompt: The system prompt to test
        n_trials: Number of attempts to make

    Returns:
        Success rate (0.0 to 1.0)
    """
    success = 0

    for _ in range(n_trials):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": planets_txt},
            ],
        )

        if is_json(response.choices[0].message.content):
            success += 1

    return success


# %% [markdown]
# Try different prompts! Here are some ideas to start with:
# - A simple "Convert to JSON" prompt
# - A detailed prompt with specific formatting instructions
# - A prompt that includes an example
# - A prompt that focuses on data types
#
# What success rates do you get? Why do you think some prompts work better than others?
#
# Example usage:
# ```python
# basic_prompt = "Convert the following into JSON format"
# success_total = test_prompt_reliability(basic_prompt)
# print(success_total)
# ```
#
# ⚠️ Note: You might notice that even with your best prompt, getting consistent results is challenging.
# This will lead us to explore more reliable techniques in the next sections!

# %% [markdown]
# ## 3. Using Built-in JSON Response Format
# Some LLM APIs provide built-in JSON formatting. Let's try it:

# %%
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": planets_txt},
    ],
    response_format={"type": "json_object"},
)

response_content = response.choices[0].message.content
print(response_content)

print(f"Is valid JSON? {is_json(response_content)}")

# %% [markdown]
# This sometimes fails due to a flaw in the LLM output. We need another method we can trust...

# %%
from openai import BadRequestError

n_trials = 10
success = 0
for _ in range(n_trials):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Convert the following into a JSON format:"},
                {"role": "user", "content": planets_txt},
            ],
            response_format={"type": "json_object"},
        )
        success += 1
    except BadRequestError as e:
        print(e)

print(f"Success rate: {success} / {n_trials}")

# %% [markdown]
# ## 4. Type-Safe Parsing with Pydantic
# Let's make our data more structured and type-safe using Pydantic models.
#
# In our previous JSON conversions, did you notice any issues with the data types?
# For example, fields like `has_global_magnetic_field` and `surface_pressure` were getting converted to strings
# because in the source text they appeared as "Yes", "No", or "Unknown".
#
# This is a common problem when dealing with unstructured data. We want to:
# - Convert "Yes"/"No" to proper boolean values
# - Handle "Unknown" cases with None
# - Ensure numeric values are actually numbers, not strings
# - Make the schema explicit and reusable
#
# Pydantic helps us solve these issues by:
# 1. Defining expected types for each field
# 2. Handling type conversion automatically
# 3. Supporting optional values with `| None` syntax
# 4. Validating the data structure

# %%
from pydantic import BaseModel


class Planet(BaseModel):
    name: str
    mass: float
    diameter: float
    density: float
    gravity: float
    escape_velocity: float
    rotation_period: float
    length_of_day: float
    distance_from_sun: float
    perihelion: float
    aphelion: float
    orbital_period: float
    orbital_velocity: float
    orbital_inclination: float
    orbital_eccentricity: float
    obliquity_to_orbit: float
    mean_temperature: float
    surface_pressure: float | None
    number_of_moons: int
    has_ring_system: str
    has_global_magnetic_field: bool | None


# %% [markdown]
# Notice how we use `float | None` and `bool | None` for fields that might be unknown.
# This tells Pydantic to:
# - Convert the field to float/bool when possible
# - Use None when the value is "Unknown" or invalid
#
# ⚠️ Always inspect your parsed data! Don't assume the LLM correctly interpreted all fields.
# Compare the output with the source text to ensure accuracy.


# %%
class SolarSystem(BaseModel):
    planets: list[Planet]


# %%
# Switching to OpenAI's API for Pydantic support
client = OpenAI(
    base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-4o-mini"

response = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": planets_txt},
    ],
    response_format=SolarSystem,
)

response_content = response.choices[0].message.content
print(response_content)


# %% [markdown]
# ### Exercise 3
# Before looking at the solution, try to define a Pydantic model for the satellites data.
# What fields would you include?

# %% [markdown]
# ### Solution


# %%
class Satellite(BaseModel):
    planet: str
    name: str
    gm: float
    radius: float
    density: float
    magnitude: float
    albedo: float


class SatelliteSystem(BaseModel):
    satellites: list[Satellite]


response = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": satellites_txt},
    ],
    response_format=SatelliteSystem,
)

response_content = response.choices[0].message.content
print(response_content)

# %% [markdown]
# Now let's combine both models:


# %%
class MilkyWay(BaseModel):
    planets: list[Planet]
    satellites: list[Satellite]


response = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": planets_txt},
        {"role": "user", "content": satellites_txt},
    ],
    response_format=MilkyWay,
)

response_content = response.choices[0].message.content
print(response_content)

# %% [markdown]
# ## 5. Understanding Token Probabilities
# Let's explore how the model makes decisions by looking at token probabilities:

# %%
# First, let's define a function to plot the top token probabilities

import matplotlib.pyplot as plt


def plot_token_probs(tokens: list[str], probs: list[float]):
    # Print probabilities
    print("Top tokens the model might generate:")
    for i, (prob, token) in enumerate(zip(probs, tokens), start=1):
        print(f"{i}.\t{token!r}\t({prob:.2%})")

    # Visualize probabilities
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(tokens)), probs)
    plt.xticks(
        ticks=range(len(tokens)),
        labels=[repr(token) for token in tokens],
        rotation=45,
        ha="right",
    )
    plt.title("Top Token Probabilities")
    plt.ylabel("Probability")
    plt.xlabel("Token")

    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


# %%
# Now we access the token probabilities from the model completion

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Convert the following into a JSON format:"},
        {"role": "user", "content": planets_txt},
    ],
    logprobs=True,
    top_logprobs=10,
)

# Extract tokens and probabilities
tokens = [lp.token for lp in response.choices[0].logprobs.content[0].top_logprobs]
probs = [10**lp.logprob for lp in response.choices[0].logprobs.content[0].top_logprobs]

# Plot and print in one call
plot_token_probs(tokens, probs)

# %% [markdown]
# ### Constraining the Output
# Sometimes we want more control over what tokens the model can generate.
# We'll explore this using a simple example: forcing the model to answer only "yes" or "no".
#
# To have more control, we will load a model and run inference locally.

# %%
# First, let's set up our model:

MODEL_NAME = "openai-community/gpt2-xl"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# %% [markdown]
# Let's see how the model responds normally to our question:

# %%
PROMPT = "Is Pluto a planet?"

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

import torch

with torch.inference_mode():
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)

answer = tokenizer.decode(outputs[0])[len(PROMPT) :].strip()
print(answer)

# %% [markdown]
# But is it consistent?

# %%
for i in range(10):
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.5,
        )

    answer = tokenizer.decode(outputs[0])[len(PROMPT) :]
    answer_fmt = answer.replace("\n", " ").strip()
    print(f"Attempt {i+1}:\t{answer_fmt}")

# %% [markdown]
# Not really... we need to find another way.

# %% [markdown]
# ### Understanding Token Probabilities
# Before we constrain the output, let's look at what tokens the model considers:

# %%
with torch.inference_mode():
    outputs = model(**inputs)
    logits = outputs.logits

last_token_logit = logits[:, -1, :]
next_token_probs = torch.nn.functional.softmax(last_token_logit, dim=-1)

k = 10
top_k_probs, top_k_indices = torch.topk(next_token_probs, k, dim=-1)
top_k_tokens = [
    tokenizer.decode(idx, skip_special_tokens=True) for idx in top_k_indices[0]
]

plot_token_probs(top_k_tokens, top_k_probs[0].tolist())

# %% [markdown]
# ### Constraining the Output
# Now, let's use a LogitsProcessor to force the model to choose between "yes" and "no":

# %%
from transformers import LogitsProcessor


class YesNoLogitsProcessor(LogitsProcessor):
    """Forces the model to output either 'yes' or 'no'."""

    def __init__(self, tokenizer, initial_length):
        self.tokenizer = tokenizer
        self.initial_length = initial_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # If we already generated a response, mask everything
        if len(input_ids[0]) > self.initial_length:
            scores.fill_(-float("inf"))
            return scores

        # Get the token IDs for "yes" and "no" from the tokenizer vocabulary
        yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)
        no_token_id = self.tokenizer.encode("no", add_special_tokens=False)

        print(f"{yes_token_id=}")
        print(f"{no_token_id=}")

        # Access the logits for the "yes" and "no" tokens from the model output
        yes_no_logits = scores[:, [yes_token_id[0], no_token_id[0]]]
        print(f"{yes_no_logits=}")

        # Convert logits to probabilities
        yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)
        print(f"{yes_no_probs=}")

        yes_prob = yes_no_probs[:, 0]
        no_prob = yes_no_probs[:, 1]

        # Set all scores to -inf
        scores.fill_(-float("inf"))

        # Set the scores for "yes" and "no" tokens to the probabilities
        scores[:, yes_token_id[0]] = torch.where(
            yes_prob > no_prob,
            input=torch.tensor(float("inf")),
            other=torch.tensor(-float("inf")),
        )
        scores[:, no_token_id[0]] = torch.where(
            yes_prob <= no_prob,
            input=torch.tensor(float("inf")),
            other=torch.tensor(-float("inf")),
        )

        return scores


# Run the constrained generation
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
prompt_tokens_length = len(inputs[0])

logits_processor = YesNoLogitsProcessor(tokenizer, prompt_tokens_length)
outputs = model.generate(
    **inputs,
    pad_token_id=tokenizer.eos_token_id,
    logits_processor=[logits_processor],
    max_length=prompt_tokens_length + 1,  # generate 1 token only
)

output_token = outputs[0, prompt_tokens_length:]
output_decoded = tokenizer.decode(output_token, skip_special_tokens=True)
print(output_decoded)

# %% [markdown]
# ### Exercise 4
# Now that you've seen how the YesNoLogitsProcessor works, how would you modify it for different binary choices?
# For example:
# - true/false responses
# - positive/negative sentiment
# - agree/disagree answers
#
# Try to sketch out the changes needed - what would you need to modify in the processor?
#
# Key points to consider:
# - How would you change the token IDs being used?
# - Would you need to modify the probability comparison?
# - What other modifications might be needed for your specific use case?

# %% [markdown]
# ## 7. Structured Generation with Outlines
# Finally, let's look at how the Outlines library makes structured generation easier.
# We'll explore three approaches to defining choices:
# 1. Using a simple list of options
# 2. Using an Enum for type-safe choices
# 3. Using a more complex structure, a Pydantic model

# %% [markdown]
# ### 7.1 List-based Choices
# The simplest way to constrain outputs is with a list of allowed values:

# %%
import outlines

outlines_model = outlines.models.transformers(MODEL_NAME)

context = """
Saturn is the sixth planet from the Sun and is best known for its spectacular ring system, which is the most extensive of any planet in our solar system.
The rings are made up of countless small particles of ice and rock, creating a stunning visual display.
Other planets in our solar system, such as Jupiter, Uranus, and Neptune, also have ring systems, but none are as prominent or extensive as Saturn's.
"""

prompt = f"""
Based on the following text, which of the following planets has the most extensive ring system?

Text: {context}
"""

# Simple list of choices
generator = outlines.generate.choice(outlines_model, ["Jupiter", "Saturn", "Uranus", "Neptune"])
answer = generator(prompt)
print(f"List-based choice: {answer}")

# %% [markdown]
# ### 7.2 Enum-based Choices
# For more complex applications, we can use Enums to:
# - Ensure type safety
# - Make choices self-documenting
# - Group related options
# - Enable IDE autocompletion

# %%
from enum import Enum


class CelestialBody(str, Enum):
    PLANET = "planet"
    STAR = "star"
    MOON = "moon"
    ASTEROID = "asteroid"
    COMET = "comet"


context = """
The Sun is a massive, luminous sphere of hot plasma that generates energy through nuclear fusion.
It is the central object in our solar system and provides the necessary light and heat for life on Earth.
"""

prompt = f"""
Based on the following text, what type of celestial body is being described?

Text: {context}
"""

# Enum-based choices
generator = outlines.generate.choice(outlines_model, CelestialBody)
answer = generator(prompt)
print(f"Enum-based choice: {answer}")

# %% [markdown]
# ### 7.3 Using Pydantic models
# We can combine Enums and Pydantic models for even more structured output:

# %%
from enum import Enum
from pydantic import BaseModel


class AtmosphereType(str, Enum):
    NONE = "none"
    THIN = "thin"
    THICK = "thick"
    DENSE = "dense"


class SurfaceType(str, Enum):
    ROCKY = "rocky"
    ICY = "icy"
    GASEOUS = "gaseous"
    METALLIC = "metallic"
    VOLCANIC = "volcanic"


class Atmosphere(BaseModel):
    type_: AtmosphereType
    main_component: str
    has_clouds: bool
    pressure_bars: float


class CelestialObject(BaseModel):
    name: str
    type_: CelestialBody  # Using our previously defined CelestialBody enum
    diameter_km: float
    surface: SurfaceType
    atmosphere: Atmosphere | None
    number_of_satellites: int


# Create a generator for this complex structure
generator = outlines.generate.json(outlines_model, CelestialObject)

context = """
Mars is the fourth planet from the Sun and the second-smallest planet (d = 6,779 km) in the Solar System, after Mercury.
It is a rocky planet with a thin atmosphere composed mainly of carbon dioxide.
Mars has two small moons, Phobos and Deimos, which are thought to be captured asteroids.
"""

prompt = f"""
Based on the following text, convert the information into a structured format:

Text: {context}
"""

celestial_object = generator(prompt)
print(repr(celestial_object))

# %% [markdown]
# ### Exercise 5
# Create your own structured generation example using Outlines.
# You can choose any of these approaches:
#
# 1. Simple list-based choices:
# ```python
# choices = ["option1", "option2", "option3"]
# generator = outlines.generate.choice(outlines_model, choices)
# ```
#
# 2. Enum-based choices:
# ```python
# class YourChoices(str, Enum):
#     CHOICE1 = "value1"
#     CHOICE2 = "value2"
# generator = outlines.generate.choice(outlines_model, YourChoices)
# ```
#
# 3. Complex nested models:
# ```python
# class SubModel(BaseModel):
#     field1: str
#     field2: YourChoices  # Using an Enum
#
# class MainModel(BaseModel):
#     name: str
#     sub_data: SubModel
#
# generator = outlines.generate.json(outlines_model, MainModel)
# ```
#
# Think about which approach better suits your use case:
# - Use lists for simple, one-off choices
# - Use enums for reusable, type-safe choices
# - Use nested models for complex, structured data
#
# Some ideas:
# - Astronomical object classification
# - Weather report structuring
# - Character/Species description
# - Scientific observation recording

# %% [markdown]
# ## Workshop Summary
# We've covered:
# 1. Converting unstructured text to JSON using LLMs
# 2. Working with different LLM providers
# 3. Using Pydantic for type-safe parsing
# 4. Understanding and controlling token generation
# 5. Using specialized libraries for structured generation
