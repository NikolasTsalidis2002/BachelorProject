


import matplotlib.pyplot as plt
import numpy as np
import json
import imageio
import os
import time
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import math
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from src.utils.utils import pdf


##########################################
#### VISUALISATION PROCEDURE ####
##########################################


##########################################
#### COLOR UTILS ####
##########################################
COLORS =["khaki", 'orangered', "khaki", "lightsteelblue", 'yellowgreen']
# "khaki", "coral"


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors."""
    color1_rgb = to_rgb(color1)
    color2_rgb = to_rgb(color2)
    return [(1-factor)*c1 + factor*c2 for c1, c2 in zip(color1_rgb, color2_rgb)]

def state_to_color(state):
    """Convert state value to an interpolated color."""
    if type(state) == int:
        return COLORS[state % len(COLORS)]
    elif type(state) == float:
        if state<0:
            return COLORS[0]
        else:
            return COLORS[1]
    elif type(state) == str:
        print("BEWARE State is a string here, convert it to int to get color variations.")
        return COLORS[0]

    else:
        return "grey"
        


##########################################
#### PLOT GRID  ####
##########################################


def plot_grid(config, data, title="", output_file="", with_llm=False, with_legend=True):

    fig, ax = plt.subplots(figsize=(8, 8))  # You can adjust the figure size as needed

    #NOTE: Because data json has bene stringified because tuple #TODO
    data_grid_tuple = {tuple(map(int, key.strip('()').split(','))): val for key, val in data.items()}

    radius=50 if config["grid"]["dimensions"][0]>30 else 200

    for key, state in data_grid_tuple.items(): #data is dictionary
        ax.scatter(key[0]+0.5, key[1]+0.5, color=state_to_color(state), s=radius)  # s controls the size of the dots
    
    # A more beautiful title
    ax.set_title(title, fontsize=15,  pad=20) #fontstyle='italic',
    
    ax.set_xlim([0, config["grid"]["dimensions"][0]])
    ax.set_ylim([0, config["grid"]["dimensions"][1]])
    ax.set_xticks([])
    ax.set_yticks([])


    # Change spine color and width outter edge
    for spine in ax.spines.values():
        spine.set_edgecolor('lavender')
        spine.set_linewidth(2)  # Adjust for desired thickness
        
    plt.tight_layout()  # To ensure the title and plots don't overlap
    plt.savefig(output_file)
    plt.close()

def plot_base(data, title="", y_label="",x_label="", output_file=""):

    # Plotting
    plt.figure(figsize=(8,6))

    # Plotting the curve
    plt.plot(data, marker='o', color='khaki', linestyle='-', linewidth=1.5)

    # Highlighting the points
    plt.scatter(range(len(data)), data, color='orangered', s=60, label='Data Points')

    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.set_xticks(range(len(data)))

    plt.savefig(output_file)
    plt.close()


##########################################
#### ANIMATE GRID  ####
##########################################


def generate_gif_from_data_grid(config, data_file="data.json", output_file="", title="", with_llm=False):

    with open(data_file, 'r') as f:
        data = json.load(f)

    images_path = []

    # Generate a plot for each time step
    for key, data_str in data.items():
        plot_grid(config, data_str, title=title+"\n t=" + str(key), output_file=output_file+f"_tp_{key}"+".png", with_llm=with_llm, with_legend=False)
        path=output_file+f"_tp_{key}"+".png"
        images_path.append(path)

    num_extra_iter=math.floor(config["n_iterations"]/config["save_every"])-len(list(data.keys())) #because of early stopping
    for _ in range(num_extra_iter):
        images_path.append(path)

    # Generate the GIF
    with imageio.get_writer(output_file+".gif", mode='I', duration=0.6) as writer:
        for image_path in images_path:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    time.sleep(5)

    # Cleanup temp images
    for img_path in images_path:
        if os.path.exists(img_path):
            os.remove(img_path)

    

##########################################
#### CONCATENATE PROCEDURES ####
##########################################

def concatenate_gifs(gif_paths=None, folder=None, output_path=None, spacing=10, contains=None, title=None):

    if gif_paths is None:
        gif_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".gif")]
        if contains is not None:
            gif_paths = [path for path in gif_paths if contains in path]

    frames_per_gif = []

    if len(gif_paths) == 0:
        return None
    # Extract all frames from each GIF
    for gif_path in gif_paths:
        gif = Image.open(gif_path)
        frames_per_gif.append([frame.copy() for frame in ImageSequence.Iterator(gif)])

    # Determine the GIF with the maximum number of frames
    max_frames = max(len(frames) for frames in frames_per_gif)

    # Ensure all GIFs have the same number of frames
    for frames in frames_per_gif:
        while len(frames) < max_frames:
            frames.append(frames[-1])

    # Assuming all GIFs have the same height
    total_width = sum([frames[0].width for frames in frames_per_gif]) + spacing * (len(gif_paths) - 1)
    height = frames_per_gif[0][0].height

    concatenated_frames = []

    for i in range(max_frames):
        new_frame = Image.new("RGB", (total_width, height), "white")
        x_offset = 0
        for j in range(len(gif_paths)):
            new_frame.paste(frames_per_gif[j][i], (x_offset, 0))
            x_offset += frames_per_gif[j][i].width + spacing

        concatenated_frames.append(new_frame)

    concatenated_frames[0].save(output_path, save_all=True, append_images=concatenated_frames[1:], duration=frames_per_gif[0][0].info['duration'], loop=0)

    return None

def concatenate_pngs(png_paths=None, folder=None, output_path=None, spacing=10, title=None, contains=None):
    
    # Get all png files in the folder
    if png_paths is None:
        png_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".png")]
        if contains is not None:
            png_paths = [path for path in png_paths if contains in path]

    print("TP", png_paths[0])
    images = [Image.open(png) for png in png_paths]

    total_width = sum([img.width for img in images]) + spacing * (len(images) - 1)
    height = images[0].height

    if title:
        # Add space for the title at the top
        # font = ImageFont.truetype("arial.ttf", 24)  # Adjust font and size as needed
        text_width, text_height = 1,1#font.getsize(title) #TODO: CHECL
        height += text_height + spacing  # Add space for title + a little more for spacing

    new_image = Image.new("RGB", (total_width, height), "white")

    if title:
        draw = ImageDraw.Draw(new_image)
        draw.text(((total_width - text_width) // 2, spacing), title,  fill="black")#font=font,

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, height - img.height))
        x_offset += img.width + spacing

    new_image.save(output_path) 


##########################################
#### PLOT SCORE ####
##########################################

def plot_final_score(score, y_label="",x_label="", output_file=""):
    #TODO: ADD VARIANCE
    fig, ax = plt.subplots()
    plt.plot(score.keys(), score.values(), 'ro')
    ax.set_title('Mean Score', fontsize=15) #TODO score name
    ax.set_xlim([0, 1]) #TODO: or depends param measure--
    ax.set_ylim([0, 1.1]) 
    ax.set_xlabel(x_label) #TODO Name parameter make vary
    ax.set_ylabel(y_label)
    if not output_file.endswith(".png"):
        output_file+=".png"
    plt.savefig(output_file)
    plt.close()


##########################################
#### PLOT DISTRIBUTION ####
##########################################
def plot_distribution_hist(data,  mu, std, scale=[1,1], path=None, xlabel="", ylabel='Number of Agents', title=""):
    
    # Plotting
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=np.arange(-3.5, 4.5, 1), align='mid', rwidth=0.7, color='olive', alpha=0.7, label="Population")
    
    # visualize the training data
    bins = np.linspace(-3.5,4.5,100)
    #multiplying to scale approximately
    plt.plot(bins, pdf(bins, mu[0], std[1], scale[0]), color='orange', label="True pdf")
    plt.plot(bins, pdf(bins, mu[1], std[0], scale[1]), color='orange')
    plt.xticks(range(-3, 4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')

    plt.savefig(path, dpi=300)
    plt.close()
