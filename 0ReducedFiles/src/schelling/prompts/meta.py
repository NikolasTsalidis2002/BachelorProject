###### PERCEIVE ########
PERCEPTION={
    "local": """Your neighborhood is composed of the following: """,
    "local_empty": """You are currently living relatively isolated: no one is living close.""",
    "local_neighbors": """{name}, which is {message} .""",
}

###### UPDATE ########
UPDATE="""Reflect upon this context, to see if {name} feels comfortable in this neighborhood. 
You ({name}) can decide either to move to another neighborhood, or to stay in this current neighborhood. 
However, keep in mind that relocating require some time, money and effort on your part. You have to balance out comfort and effort realistically.
Please respond with "MOVE" if {name} wish to change your neighborhood, or "STAY" if {name} prefer to remain in the neighborhood."""

META_PROMPTS={
    "perception": PERCEPTION,
    "update": UPDATE
}


#In a few months, you can decide..
# However, remember that relocating will require some time and effort on your part.
#TODO: Compare "Next year", "In a few months", "..."



