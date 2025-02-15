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
#     display_name: .venv
#     language: python
#     name: python3
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

import os

from dotenv import load_dotenv

# %%
from openai import OpenAI

load_dotenv()

# We'll start with Groq's API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

# %% [markdown]
# ### Exercise 1
# Try to write a prompt that would convert the planets data into JSON format.
# What challenges do you expect to face?

# %%
PROMPT = ...  # fill this in

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": PROMPT},
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
# Let's test it with a basic prompt:

# %%
PROMPT = "Convert the following into a JSON format:"
success = test_prompt_reliability(PROMPT, n_trials=1)

print(f"Conversion successful: {success == 1}")

# %% [markdown]
# Try different prompts! Here are some ideas to start with:
# - A simple "Convert to JSON" prompt
# - A detailed prompt with specific formatting instructions
# - A prompt that includes an example
# - A prompt that focuses on data types
#
# What success rates do you get? Why do you think some prompts work better than others?
#
# ⚠️ Note: You might notice that even with your best prompt, getting consistent results is challenging.
# This will lead us to explore more reliable techniques in the next sections!

# %%
MY_PROMPT = "..."  # Fill in your prompt here
N_TRIALS = 10

success = test_prompt_reliability(MY_PROMPT, N_TRIALS)
print(f"Success rate: {success} / {N_TRIALS}")

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
                {
                    "role": "system",
                    "content": "Convert the following into a JSON format:",
                },
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
# Try to define a Pydantic model for the satellites data.
# What fields would you include?

# %%
class Satellite(BaseModel):
    name: str
    # fill in the rest of the fields


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

# Plot and print the top token probabilities
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

from transformers import AutoModelForCausalLM, AutoTokenizer

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

# %%
class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, initial_length):
        self.tokenizer = tokenizer
        self.initial_length = initial_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Your code here
        return scores


# %%
MY_PROMPT = "..."  # Fill in your prompt here

# Run the constrained generation
inputs = tokenizer(MY_PROMPT, return_tensors="pt").to(model.device)
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
generator = outlines.generate.choice(
    outlines_model, ["Jupiter", "Saturn", "Uranus", "Neptune"]
)
answer = generator(prompt)
print(f"List-based choice: {answer}")

# %% [markdown]
# ### 7.2 Enum-based Choices
# For more complex applications, we can use Enums to:
# - Ensure type safety
# - Make choices self-documenting
# - Group related options

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
    type: AtmosphereType
    main_component: str
    has_clouds: bool
    pressure_bars: float


class CelestialObject(BaseModel):
    name: str
    type: CelestialBody  # Using our previously defined CelestialBody enum
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
# Find some unstructured data online for this. You can use the `get_text` function defined at the beginning of this notebook.
#
# Some ideas:
# - Astronomical object classification
# - Weather report structuring
# - Character/Species description
# - Scientific observation recording


# %%
class MyModel(BaseModel):
    # Fill in the fields for your custom model


# Create a generator for this complex structure

generator = outlines.generate.json(outlines_model, MyModel)

context = """
... # Fill in the context here, using get_text() or any other method
"""

prompt = f"""
Based on the following text, convert the information into a structured format:

Text: {context}
"""

my_model = generator(prompt)
print(repr(my_model))

# %% [markdown]
# ## 6. Adding Data Validation for LLM Outputs
#
# One key challenge when working with LLMs is that they can:
# 1. Generate physically impossible values
# 2. Make mathematical errors
# 3. Fail to maintain consistency between related values
# 4. Hallucinate plausible-looking but incorrect data
#
# Even with perfect LLM output, the source data itself might contain errors or inconsistencies.
# Therefore, we need robust validation to:
# - Enforce physical constraints (e.g., positive mass, speed less than light)
# - Check mathematical relationships (e.g., orbital parameters)
# - Validate consistency between related values
# - Flag impossible combinations
#
# Let's look at some examples using astronomical data, where physical constraints
# are particularly important.

# %%
# First, let's create some questionable astronomical texts for the LLM to process
questionable_star_text = """
Alpha Centauri is a remarkable star system located -4.37 light years from Earth.
The main star has a negative mass of -2.1 solar masses and a radius of 0 kilometers.
Scientists believe it might be both a neutron star and a red giant simultaneously.
"""

impossible_planetary_system = """
The Kepler-X system contains a small star with mass 1e28 kg.
It has three planets:
1. Super-Jupiter: A massive planet with mass 1e29 kg (10 times more than its star!)
2. Speedy: Completes an orbit in -2 days at a distance of 1e8 km
3. Paradox: Has a closest approach (perihelion) of 2e8 km but furthest point (aphelion) of 1e8 km
"""

# %%
# Set up our client
client = OpenAI(
    base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-4o-mini"


# First, let's see what the LLM generates without validation
class Star(BaseModel):
    name: str
    distance_ly: float
    mass_solar: float
    radius_km: float
    type: str


response = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "Convert the following star description into structured data:",
        },
        {"role": "user", "content": questionable_star_text},
    ],
    response_format=Star,
)

print("Without validation:")
print(response.choices[0].message.content)

# %%
# Now let's add validators

from pydantic import ValidationError, ValidationInfo, field_validator, model_validator


class ValidatedStar(BaseModel):
    name: str
    distance_ly: float
    mass_solar: float
    radius_km: float
    type: str

    @field_validator("distance_ly", "mass_solar", "radius_km")
    @classmethod
    def must_be_positive(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("type")
    @classmethod
    def validate_star_type(cls, value: str) -> str:
        valid_types = {
            "red dwarf",
            "red giant",
            "neutron star",
            "white dwarf",
            "main sequence",
        }
        if value.lower() not in valid_types:
            raise ValueError(f"Invalid star type. Must be one of: {valid_types}")
        return value.lower()


try:
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Convert the following star description into structured data:",
            },
            {"role": "user", "content": questionable_star_text},
        ],
        response_format=ValidatedStar,
    )
except ValidationError as e:
    print("\nWith validation:")
    print(e)

# %%
# Now let's validate a planetary system

from typing import Self


class Planet(BaseModel):
    name: str
    mass_kg: float
    orbital_period_days: float
    perihelion_km: float
    aphelion_km: float

    @field_validator("mass_kg", "orbital_period_days", "perihelion_km", "aphelion_km")
    @classmethod
    def must_be_positive(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @model_validator(mode="after")
    def check_orbit(self) -> Self:
        if self.perihelion_km >= self.aphelion_km:
            raise ValueError(
                f"Perihelion ({self.perihelion_km:e} km) must be less than "
                f"aphelion ({self.aphelion_km:e} km)"
            )
        return self


class PlanetarySystem(BaseModel):
    star_name: str
    star_mass_kg: float
    planets: list[Planet]

    @model_validator(mode="after")
    def validate_masses(self) -> Self:
        for planet in self.planets:
            if planet.mass_kg >= self.star_mass_kg:
                raise ValueError(
                    f"Planet {planet.name} has mass {planet.mass_kg:e} kg, which is "
                    f"greater than or equal to its star ({self.star_mass_kg:e} kg)"
                )
        return self


try:
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Convert this planetary system description into structured data:",
            },
            {"role": "user", "content": impossible_planetary_system},
        ],
        response_format=PlanetarySystem,
    )
except ValidationError as e:
    print("\nPlanetary system validation errors:")
    print(e)

# %% [markdown]
# ### Exercise 4.1: Black Hole Validator
# Create a validator for black hole data. The LLM might generate physically impossible values.
# Key physics to check:
# - Event horizon radius (R) = 2GM/c² (G = gravitational constant, M = mass, c = speed of light)
# - Hawking radiation temperature ∝ 1/M
# - Singularity must be within event horizon
#
# Here's some intentionally problematic text to test with:

# %%
impossible_black_hole = """
BH-123 is a unique black hole with:
- Mass: -5e30 kg (negative mass!)
- Event horizon: 100 km (too large for its mass)
- Singularity distance: 200 km (outside event horizon!)
- Hawking temperature: -290 K (negative temperature!)
"""

G = 6.674e-11  # gravitational constant
c = 3e8  # speed of light


class BlackHole(BaseModel):
    name: str
    mass_kg: float
    event_horizon_radius_km: float
    singularity_distance_km: float | None = None
    hawking_temperature_k: float | None = None

    @model_validator(mode="after")
    def validate_event_horizon(self) -> Self:
        # calculate expected radius
        expected_radius =  # fill in the formula here
        if # fill in the condition here
            raise ValueError(
                f"..."  # fill in the error message here
            )
        return self
    
    @model_validator(mode="after")
    def validate_singularity(self) -> Self:
        if self.singularity_distance_km is not None:
            # fill code here
        return self
    
    @model_validator(mode="after")
    def validate_temperature(self) -> Self:
        # fill code here


try:
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Convert this black hole description into structured data:",
            },
            {"role": "user", "content": impossible_black_hole},
        ],
        response_format=BlackHole,
    )
except ValidationError as e:
    print("\nBlack hole validation errors:")
    print(e)

# %% [markdown]
# ### Exercise 4.2: Testing with Valid Data
#
# Now that we've seen how our validators catch impossible values, let's try them with
# physically possible data. Create your own `possible_black_hole` text with realistic values.
#
# Some tips for realistic values:
# - Stellar black holes typically have masses of 3-100 solar masses (1 solar mass ≈ 2e30 kg)
# - The event horizon radius should follow r = 2GM/c²
# - Any singularity distance should be > 0 but < event horizon radius
# - Hawking temperature is inversely proportional to mass
#
# Try running your data through the validator:

# %%
possible_black_hole = """
# Fill in your own realistic black hole description here
"""

try:
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Convert this black hole description into structured data:",
            },
            {"role": "user", "content": possible_black_hole},
        ],
        response_format=BlackHole,
    )
    print("Validation successful!")
    print(response.choices[0].message.content)
except ValidationError as e:
    print("Validation failed:")
    print(e)

# %% [markdown]
# ## 8. Simple Vision Models with Structured Output
# Let's explore how to get structured output from vision models using Outlines.

import outlines

# %%
import torch
from transformers import LlavaNextForConditionalGeneration

# Initialize our model
model = outlines.models.transformers_vision(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    model_class=LlavaNextForConditionalGeneration
)

# %%
from pydantic import BaseModel


class ImageData(BaseModel):
    caption: str
    tags_list: list[str]
    object_list: list[str]
    is_photo: bool

# Create our structured generator
image_data_generator = outlines.generate.json(model, ImageData)

# %%
from io import BytesIO
from urllib.request import urlopen

from PIL import Image


def img_from_url(url: str) -> Image.Image:
    """Load an image from a URL and convert it to RGB format."""
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")

# Test with a famous image
image_url = "https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg"
image = img_from_url(image_url)

# Lower image quality for faster processing
image = image.resize((256, 256))

# Generate structured output
result = image_data_generator(
    "<image> detailed JSON metadata:",
    [image]
)
print(result)

# %% [markdown]
# ### Exercise: Testing with Different Images
# Try running the structured generation with different types of images to see how the model performs.
